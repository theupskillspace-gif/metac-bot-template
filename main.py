from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import textwrap
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from tavily import TavilyClient
from pydantic import model_validator

try:
    from forecasting_tools import (
        BinaryQuestion,
        ForecastBot,
        GeneralLlm,
        MetaculusClient,
        MetaculusQuestion,
        MultipleChoiceQuestion,
        NumericDistribution,
        NumericQuestion,
        Percentile,
        BinaryPrediction,
        PredictedOptionList,
        PredictedOption,
        ReasonedPrediction,
        clean_indents,
        structure_output,
    )
except ImportError as e:
    raise ImportError("Failed to import forecasting_tools.") from e

for name in ["NumericQuestion", "BinaryQuestion", "MultipleChoiceQuestion", "PredictedOptionList"]:
    if name not in globals():
        raise NameError(f"Type '{name}' not imported.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = Live = Panel = Text = lambda *args, **kwargs: None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger("UpskillBot")

DEFAULT_FORECASTER = "openrouter/anthropic/claude-sonnet-4.5"
PARSER_MODEL = "openrouter/openai/gpt-4.1-mini"
SUMMARIZER_MODEL = "openrouter/openai/gpt-4.1-mini"

@model_validator(mode="after")
def _normalize_probs(self: PredictedOptionList):
    if not getattr(self, "predicted_options", None):
        return self
    probs = [float(p.probability) for p in self.predicted_options]
    total = sum(probs)
    if total <= 0:
        logger.warning(f"PredictedOptionList sum is {total}. Raw: {self.predicted_options}")
        return self
    if total > 1.5:
        for opt in self.predicted_options:
            opt.probability = float(opt.probability) / 100.0
        probs = [float(p.probability) for p in self.predicted_options]
        total = sum(probs)
        if total <= 0:
            return self
    if abs(total - 1.0) > 0.001:
        for opt in self.predicted_options:
            opt.probability = float(opt.probability) / total
    for opt in self.predicted_options:
        opt.probability = max(0.0, min(1.0, float(opt.probability)))
    total2 = sum(float(p.probability) for p in self.predicted_options)
    if total2 > 0 and abs(total2 - 1.0) > 1e-6:
        for opt in self.predicted_options:
            opt.probability = float(opt.probability) / total2
    return self

PredictedOptionList.__pydantic_post_validate__ = _normalize_probs
logger.info("✅ Monkey-patched PredictedOptionList validator.")

def median(lst: List[float]) -> float:
    if not lst:
        raise ValueError("median() arg is an empty sequence")
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0
    return float(sorted_lst[mid])

def build_tavily_query(question: MetaculusQuestion, max_chars: int = 300) -> str:
    q = question.question_text.strip()
    bg = (question.background_info or "").strip()
    q = re.sub(r"http\S+", "", q)
    bg = re.sub(r"http\S+", "", bg)
    q = re.sub(r"\s+", " ", q).strip()
    bg = re.sub(r"\s+", " ", bg).strip()

    if len(q) <= max_chars:
        if not bg:
            return q
        candidate = f"{q} — {bg}"
        if len(candidate) <= max_chars:
            return candidate
        space_for_bg = max_chars - len(q) - 3
        if space_for_bg > 10:
            bg_part = textwrap.shorten(bg, width=space_for_bg, placeholder="…")
            return f"{q} — {bg_part}"
        return q

    first_sent = q.split(".")[0].strip()
    if len(first_sent) > max_chars:
        return textwrap.shorten(first_sent, width=max_chars, placeholder="…")

    remaining = max_chars - len(first_sent) - 3
    if remaining > 10 and bg:
        bg_part = textwrap.shorten(bg, width=remaining, placeholder="…")
        combo = f"{first_sent} — {bg_part}"
        if len(combo) <= max_chars:
            return combo

    return textwrap.shorten(q, width=max_chars, placeholder="…")

def strict_truncate_query(base: str, suffix: str = "", max_len: int = 395) -> str:
    full = f"{base} {suffix}".strip()
    if len(full) <= max_len:
        return full
    available = max_len - len(suffix) - 1
    if available <= 0:
        return textwrap.shorten(suffix, width=max_len, placeholder="…")
    truncated_base = textwrap.shorten(base, width=available, placeholder="…")
    result = f"{truncated_base} {suffix}".strip()
    return result[:max_len]

class UpskillBot(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY must be set.")
        self.tavily = TavilyClient(api_key=api_key)
        self._tavily_query_count = 0
        self._max_tavily_queries = 400
        self._tavily_lock = asyncio.Lock()
        self._prediction_records: List[Dict[str, Any]] = []
        self._research_cache: Dict[str, str] = {}

        self._cost_tracker = {}
        self._model_pricing = {
            "gpt-4.1-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        }
        self._encoding_cache = {}

        self._console = Console() if RICH_AVAILABLE else None
        self._live_display = None
        self._questions_processed = 0
        self._questions_total = 0

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": DEFAULT_FORECASTER,
            "parser": PARSER_MODEL,
            "researcher_claude": DEFAULT_FORECASTER,
            "summarizer": SUMMARIZER_MODEL,
        }

    def _get_encoding(self, model_name: str):
        if not TIKTOKEN_AVAILABLE:
            return None
        if model_name in self._encoding_cache:
            return self._encoding_cache[model_name]
        enc = tiktoken.get_encoding("cl100k_base")
        self._encoding_cache[model_name] = enc
        return enc

    def _estimate_cost(self, model_path: str, prompt: str, completion: str) -> float:
        if not TIKTOKEN_AVAILABLE:
            return 0.0
        if model_path not in self._cost_tracker:
            self._cost_tracker[model_path] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

        model_key = model_path.split("/")[-1]
        pricing = self._model_pricing.get(model_key, {"input": 1.0, "output": 3.0})
        enc = self._get_encoding(model_key)
        if not enc:
            return 0.0

        input_tokens = len(enc.encode(prompt))
        output_tokens = len(enc.encode(completion))

        self._cost_tracker[model_path]["input_tokens"] += input_tokens
        self._cost_tracker[model_path]["output_tokens"] += output_tokens
        self._cost_tracker[model_path]["calls"] += 1

        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    async def _invoke_with_cost_tracking(self, model_name: str, prompt: str) -> str:
        llm = self.get_llm(model_name, "llm")
        response = await llm.invoke(prompt)
        self._estimate_cost(llm.model, prompt, response)
        return response

    async def _tavily_search_limited(self, query: str, **kwargs) -> dict:
        async with self._tavily_lock:
            if self._tavily_query_count >= self._max_tavily_queries:
                raise RuntimeError(f"UpskillBot: Tavily limit ({self._max_tavily_queries}) reached.")
            self._tavily_query_count += 1
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.tavily.search(query=query.strip(), **kwargs))

    def _is_stock_question(self, question: MetaculusQuestion) -> bool:
        text = " ".join([question.question_text, question.background_info or "", question.resolution_criteria or ""]).lower()
        patterns = [r"\b(?:stock|equity|share|s&p|nasdaq|dow|ticker)\b", r"\b\$?[a-z]{1,5}\b"]
        return any(re.search(pat, text) for pat in patterns)

    def _estimate_question_difficulty(self, question: MetaculusQuestion) -> float:
        text = (question.question_text + " " + (question.background_info or "")).lower()
        now = datetime.now(timezone.utc)
        days_to_close = (question.close_time - now).total_seconds() / 86400 if question.close_time else 365
        base_rate_hint = any(w in text for w in ["rare", "unlikely", "first time", "never before", "unprecedented"])
        long_horizon = days_to_close > 365
        vague_resolution = "ambiguous" in (question.resolution_criteria or "").lower()
        return min(1.0, 0.3 + 0.3 * long_horizon + 0.2 * base_rate_hint + 0.2 * vague_resolution)

    def _get_numeric_median(self, dist: NumericDistribution) -> float:
        for p in dist.declared_percentiles:
            if abs(p.percentile - 0.5) < 0.01 or abs(p.percentile - 50.0) < 1.0:
                return float(p.value)
        sorted_pts = sorted(dist.declared_percentiles, key=lambda x: x.percentile)
        if not sorted_pts:
            return 0.0
        normalized = []
        for pt in sorted_pts:
            perc = pt.percentile / 100.0 if pt.percentile > 1.0 else pt.percentile
            normalized.append(Percentile(percentile=perc, value=pt.value))
        if len(normalized) == 1:
            return float(normalized[0].value)
        if normalized[0].percentile >= 0.5:
            return float(normalized[0].value)
        if normalized[-1].percentile <= 0.5:
            return float(normalized[-1].value)
        for i in range(len(normalized) - 1):
            p1, p2 = normalized[i], normalized[i + 1]
            if p1.percentile <= 0.5 <= p2.percentile:
                frac = (0.5 - p1.percentile) / (p2.percentile - p1.percentile)
                return float(p1.value + frac * (p2.value - p1.value))
        return float(normalized[-1].value)

    def _interpolate_percentile(self, percentiles: List[Percentile], target_p: float) -> float:
        sorted_pts = sorted(percentiles, key=lambda x: x.percentile)
        if not sorted_pts:
            return 0.0
        if target_p <= sorted_pts[0].percentile:
            return float(sorted_pts[0].value)
        if target_p >= sorted_pts[-1].percentile:
            return float(sorted_pts[-1].value)
        for i in range(len(sorted_pts) - 1):
            p1, p2 = sorted_pts[i], sorted_pts[i + 1]
            if p1.percentile <= target_p <= p2.percentile:
                frac = (target_p - p1.percentile) / (p2.percentile - p1.percentile)
                return float(p1.value + frac * (p2.value - p1.value))
        return float(sorted_pts[-1].value)

    async def run_research(self, question: MetaculusQuestion) -> str:
        qid = getattr(question, "id", getattr(question, "question_id", hash(question.question_text)))
        cache_key = str(qid)
        if cache_key in self._research_cache:
            return self._research_cache[cache_key]

        async with self._concurrency_limiter:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            base_query = build_tavily_query(question)

            recent_summary = "[Recent developments pending]"
            try:
                recent_query = strict_truncate_query(base_query, "Focus on developments in the last 6 months.", 395)
                recent = await self._tavily_search_limited(
                    recent_query,
                    search_depth="advanced",
                    max_results=4,
                    days=180,
                )
                snippets = [
                    f"[{i+1}] {r.get('title','')}: {textwrap.shorten(r.get('content',''), width=180, placeholder='…')}"
                    for i, r in enumerate(recent.get("results", [])[:4])
                ]
                recent_summary = "\n".join(snippets) if snippets else "[No recent results]"
            except Exception as e:
                logger.error(f"Recent Tavily failed: {e}")
                recent_summary = f"[Error: {e}]"

            historical_summary = "[Historical trends pending]"
            try:
                historical_query = strict_truncate_query(base_query, "Historical base rates, long-term trends, reference class.", 395)
                historical = await self._tavily_search_limited(
                    historical_query,
                    search_depth="advanced",
                    max_results=4,
                )
                snippets = [
                    f"[{i+1}] {r.get('title','')}: {textwrap.shorten(r.get('content',''), width=180, placeholder='…')}"
                    for i, r in enumerate(historical.get("results", [])[:4])
                ]
                historical_summary = "\n".join(snippets) if snippets else "[No historical data]"
            except Exception as e:
                logger.error(f"Historical Tavily failed: {e}")
                historical_summary = f"[Error: {e}]"

            claude_prompt = clean_indents(f"""
                You are a Good Judgment Project-style forecaster. Be calibrated and evidence-based.
                Today (UTC): {today_str}

                QUESTION:
                {question.question_text}

                BACKGROUND:
                {question.background_info or 'None'}

                RESOLUTION CRITERIA:
                {question.resolution_criteria or 'None'}

                RECENT (last 6 months) SNIPPETS:
                {recent_summary}

                HISTORICAL / BASE RATE SNIPPETS:
                {historical_summary}

                Output strictly in this structure:

                Base rate (outside view): <one numeric probability or percent with 1–2 sentences>
                Key uncertainties/drivers (3–6 bullets): <bullets>
                Signposts to watch (3 bullets): <bullets>
                Common failure modes: <2 bullets>
            """)
            claude_response = await self._invoke_with_cost_tracking("default", claude_prompt)

            full_research = clean_indents(
                f"""
                ### UpskillBot Research (as of {today_str})

                --- OUTSIDE VIEW / DRIVERS / SIGNPOSTS ---
                {claude_response}

                --- RECENT DEVELOPMENTS (last 6mo) ---
                {recent_summary}

                --- HISTORICAL TRENDS / BASE RATE ---
                {historical_summary}
                """
            )
            self._research_cache[cache_key] = full_research
            return full_research

    def _record_prediction(
        self,
        question: MetaculusQuestion,
        prob: Optional[float],
        reasoning: str,
        extra: Optional[Dict] = None,
    ):
        try:
            qid = getattr(question, "id", None)
            if qid is None:
                qid = getattr(question, "question_id", f"anon_{hash(question.question_text) % 10000}")

            record = {
                "question_id": qid,
                "page_url": getattr(question, "page_url", "N/A"),
                "title": getattr(question, "question_text", "Unknown Question")[:100],
                "type": question.__class__.__name__,
                "predicted_prob": prob,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "tavily_queries_used": self._tavily_query_count,
                "is_stock": self._is_stock_question(question),
                "difficulty_score": self._estimate_question_difficulty(question),
                "reasoning_snippet": reasoning[:500].replace("\n", " "),
            }
            if extra:
                safe_extra = {}
                for k, v in extra.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        safe_extra[k] = v
                    else:
                        try:
                            safe_extra[k] = json.dumps(v, ensure_ascii=False)
                        except Exception:
                            safe_extra[k] = str(v)
                record.update(safe_extra)
            self._prediction_records.append(record)
        except Exception as e:
            logger.debug(f"Non-fatal: Skipped recording prediction ({e})")

    async def _run_forecast_with_red_team(
        self, question: MetaculusQuestion, research: str, is_binary: bool = True
    ) -> Tuple[str, float]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        initial_prompt = clean_indents(f"""
            You are a superforecaster trained on the Good Judgment Project. Today (UTC): {today}.
            You must be calibrated: avoid overconfidence; use 1%–99% not 0/100.
            Prefer outside view first, then inside view adjustments.

            QUESTION:
            {question.question_text}

            BACKGROUND:
            {question.background_info or 'None'}

            RESOLUTION CRITERIA:
            {question.resolution_criteria or 'None'}

            RESEARCH:
            {research}

            METHOD:
            1) Outside view: propose a reference class and a base-rate probability.
            2) Decompose into 3–6 drivers. For each, state direction of effect and a rough conditional adjustment.
            3) Consider at least 2 plausible worlds (pro and con) and give a "pre-mortem" on failure modes.
            4) Provide a 10th and 90th percentile range for the final probability before choosing the final number.
            5) If evidence is weak/contradictory, shrink toward 50%.

            OUTPUT:
            - Brief reasoning (high-signal)
            - Final line exactly: Probability: ZZ%
        """)
        initial_reasoning = await self._invoke_with_cost_tracking("default", initial_prompt)

        red_team_prompt = clean_indents(f"""
            You are a skeptical expert trying to falsify the forecast below.
            Provide:
            - 3 strongest counterarguments
            - 2 concrete signposts that would indicate the forecast is off
            - 1 alternative reference class/base-rate framing that points to a different probability
            - A calibration critique: is the probability too extreme given the evidence?

            FORECAST TO CRITIQUE:
            {initial_reasoning}

            CONTEXT:
            {question.question_text}
            {question.background_info or ''}
            {question.resolution_criteria or ''}

            RESEARCH:
            {research}
        """)
        red_team_response = await self._invoke_with_cost_tracking("default", red_team_prompt)

        final_prompt = clean_indents(f"""
            You will revise the forecast if warranted using the critique.
            Rules:
            - Maintain calibration (avoid unjustified extremes).
            - Make the minimum necessary update based on the critique.
            - Keep reasoning concise and causal.
            - Final line exactly: Probability: ZZ%

            ORIGINAL FORECAST:
            {initial_reasoning}

            RED TEAM CRITIQUE:
            {red_team_response}
        """)
        revised_reasoning = await self._invoke_with_cost_tracking("default", final_prompt)

        prob = 0.5
        if is_binary:
            try:
                pred: BinaryPrediction = await structure_output(
                    revised_reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
                )
                prob = max(0.01, min(0.99, float(pred.prediction_in_decimal)))
            except Exception as e:
                logger.warning(f"Parse fail during red teaming: {e}")

        return revised_reasoning, prob

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        reasoning, prob = await self._run_forecast_with_red_team(question, research, is_binary=True)
        self._record_prediction(question, prob, reasoning)
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prompt = clean_indents(f"""
            You are a calibrated superforecaster. Today (UTC): {today}.
            Use outside view + decomposition; avoid arbitrary equal-split unless truly no info.
            Probabilities must be decimals in [0,1] and sum to 1.

            QUESTION:
            {question.question_text}

            BACKGROUND:
            {question.background_info or 'None'}

            RESOLUTION CRITERIA:
            {question.resolution_criteria or 'None'}

            OPTIONS:
            {question.options}

            RESEARCH:
            {research}

            OUTPUT:
            - Brief reasoning
            - Then one line per option exactly like:
              <Option text>: 0.23
        """)
        reasoning = await self._invoke_with_cost_tracking("default", prompt)
        try:
            pred: PredictedOptionList = await structure_output(
                reasoning,
                PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=f"Options: {question.options}. Probabilities must be decimals (0-1) summing to 1.",
            )
        except Exception as e:
            logger.warning(f"MC parse fail Q{getattr(question, 'id', 'unknown')}: {e}")
            p = 1.0 / max(1, len(question.options))
            pred = PredictedOptionList(predicted_options=[PredictedOption(option_name=opt, probability=p) for opt in question.options])

        prob_dict = {opt.option_name: float(opt.probability) for opt in pred.predicted_options}
        if prob_dict:
            top_opt = max(prob_dict, key=prob_dict.get)
            top_prob = float(prob_dict[top_opt])
        else:
            top_opt = "N/A"
            top_prob = None
        self._record_prediction(question, top_prob, reasoning, extra={"top_option": top_opt})
        return ReasonedPrediction(prediction_value=pred, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lower_msg = f"Lower bound: {question.lower_bound}." if question.lower_bound is not None else "No explicit lower bound."
        upper_msg = f"Upper bound: {question.upper_bound}." if question.upper_bound is not None else "No explicit upper bound."
        bounds_msg = f"{lower_msg} {upper_msg}"

        prompt = clean_indents(f"""
            You are a calibrated quantitative forecaster. Today (UTC): {today}.
            Use outside view (reference class) first, then inside view adjustments.
            Avoid overconfidence; if uncertain, widen tails.

            QUESTION:
            {question.question_text}

            BACKGROUND:
            {question.background_info or 'None'}

            RESOLUTION CRITERIA:
            {question.resolution_criteria or 'None'}

            BOUNDS:
            {bounds_msg}

            RESEARCH:
            {research}

            TASK:
            Provide percentiles for the target distribution at p in {{0.10, 0.20, 0.40, 0.60, 0.80, 0.90}}.
            Use p as decimals (0.10 etc). Ensure values respect bounds.

            OUTPUT:
            - Brief reasoning
            - Then a JSON-like list of objects (parsable) with keys percentile and value, e.g.
              [{"percentile":0.1,"value":123}, ...]
        """)
        reasoning = await self._invoke_with_cost_tracking("default", prompt)
        try:
            pct_list: list[Percentile] = await structure_output(
                reasoning, list[Percentile], model=self.get_llm("parser", "llm")
            )
            dist = NumericDistribution.from_question(pct_list, question)
        except Exception as e:
            logger.warning(f"Numeric parse fail: {e}")
            lo = float(question.lower_bound if question.lower_bound is not None else 0.0)
            hi = float(question.upper_bound if question.upper_bound is not None else lo + 1.0)
            fallback_ps = [0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
            fallback = [Percentile(percentile=p, value=lo + (hi - lo) * p) for p in fallback_ps]
            dist = NumericDistribution.from_question(fallback, question)

        median_val = self._get_numeric_median(dist)
        self._record_prediction(question, None, reasoning, extra={"median": median_val})
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        if isinstance(question, BinaryQuestion):
            return await self._run_forecast_on_binary(question, research)
        if isinstance(question, MultipleChoiceQuestion):
            return await self._run_forecast_on_multiple_choice(question, research)
        if isinstance(question, NumericQuestion):
            return await self._run_forecast_on_numeric(question, research)
        raise ValueError(f"Unsupported: {type(question)}")

    async def _compute_brier_scores(self):
        try:
            client = MetaculusClient()
            binary_records = [
                r for r in self._prediction_records
                if r["type"] == "BinaryQuestion" and r["predicted_prob"] is not None
            ]
            question_ids = [
                r["question_id"] for r in binary_records
                if isinstance(r["question_id"], (int, str)) and r["question_id"] not in ("N/A", "unknown")
            ]
            if not question_ids:
                return
            all_qs = await client.get_questions_by_ids(question_ids)
            resolved_qs = [q for q in all_qs if isinstance(q, BinaryQuestion) and q.resolution in ("yes", "no")]
            brier_sum = log_score_sum = scored = 0.0
            for q in resolved_qs:
                rec = next((r for r in binary_records if r["question_id"] == q.id), None)
                if rec:
                    pred = float(rec["predicted_prob"])
                    actual = 1.0 if q.resolution == "yes" else 0.0
                    brier = (pred - actual) ** 2
                    eps = 1e-6
                    clipped_pred = max(eps, min(1 - eps, pred))
                    log_score = actual * math.log(clipped_pred) + (1 - actual) * math.log(1 - clipped_pred)
                    brier_sum += brier
                    log_score_sum += log_score
                    scored += 1
                    rec.update({
                        "resolution": q.resolution,
                        "actual": actual,
                        "brier_score": round(brier, 4),
                        "log_score": round(log_score, 4)
                    })
            if scored:
                logger.info(f"📊 Avg Brier (n={int(scored)}): {brier_sum / scored:.4f}")
                logger.info(f"📊 Avg Log Score (n={int(scored)}): {log_score_sum / scored:.4f}")
        except Exception as e:
            logger.error(f"Brier/log score computation failed: {e}")

    def export_predictions_to_csv(self, filepath: str = "upskill_bot_forecasts.csv"):
        if not self._prediction_records:
            return
        safe_records = []
        for r in self._prediction_records:
            safe_r = {}
            for k, v in r.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    safe_r[k] = v
                else:
                    try:
                        safe_r[k] = json.dumps(v, ensure_ascii=False)
                    except Exception:
                        safe_r[k] = str(v)
            safe_records.append(safe_r)
        df = pd.DataFrame(safe_records)
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Exported {len(df)} predictions to {filepath}")

    def export_cost_report(self, filepath: str = "upskill_bot_costs.csv"):
        if not self._cost_tracker:
            return
        records = []
        total_cost = 0.0
        for model, stats in self._cost_tracker.items():
            model_key = model.split("/")[-1]
            pricing = self._model_pricing.get(model_key, {"input": 1.0, "output": 3.0})
            cost = (stats["input_tokens"] * pricing["input"] + stats["output_tokens"] * pricing["output"]) / 1_000_000
            total_cost += cost
            records.append({
                "model": model,
                "calls": stats["calls"],
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "estimated_cost_usd": round(cost, 6)
            })
        if records:
            df = pd.DataFrame(records)
            df.to_csv(filepath, index=False)
            logger.info(f"✅ Exported cost report to {filepath}")
            logger.info(f"💰 Total estimated cost: ${total_cost:.4f}")

    def _render_dashboard(self) -> Panel:
        if not RICH_AVAILABLE:
            return Panel("Install 'rich' for live dashboard")
        progress = f"{self._questions_processed}/{self._questions_total}" if self._questions_total else "N/A"
        total_cost = 0.0
        cost_table = Table(show_header=True, header_style="bold magenta")
        cost_table.add_column("Model")
        cost_table.add_column("Calls")
        cost_table.add_column("In Tokens")
        cost_table.add_column("Out Tokens")
        cost_table.add_column("Est. Cost (USD)")
        for model, stats in self._cost_tracker.items():
            model_key = model.split("/")[-1]
            pricing = self._model_pricing.get(model_key, {"input": 1.0, "output": 3.0})
            cost = (stats["input_tokens"] * pricing["input"] + stats["output_tokens"] * pricing["output"]) / 1_000_000
            total_cost += cost
            cost_table.add_row(
                model,
                str(stats["calls"]),
                f"{stats['input_tokens']:,}",
                f"{stats['output_tokens']:,}",
                f"${cost:.4f}"
            )
        tavily_info = f"Tavily Queries: {self._tavily_query_count}/{self._max_tavily_queries}"
        content = Text.assemble(
            f"Questions Processed: {progress}\n",
            f"Total Est. Cost: ${total_cost:.4f}\n",
            tavily_info,
            "\n\n",
        )
        content.append(cost_table)
        return Panel(content, title="📈 UpskillBot Live Dashboard", border_style="green")

    async def _forecast_single_question(self, question: MetaculusQuestion):
        self._questions_processed += 1
        if self._live_display and RICH_AVAILABLE:
            self._live_display.update(self._render_dashboard())
        return await super()._forecast_single_question(question)

    async def run_all_tournaments(self, tournament_ids: List):
        if RICH_AVAILABLE:
            with Live(self._render_dashboard(), refresh_per_second=1, console=self._console) as live:
                self._live_display = live
                for tid in tournament_ids:
                    logger.info(f"▶ Forecasting tournament: {tid}")
                    await self.forecast_on_tournament(tid, return_exceptions=True)
                await self._compute_brier_scores()
                self.export_predictions_to_csv()
                self.export_cost_report()
        else:
            for tid in tournament_ids:
                logger.info(f"▶ Forecasting tournament: {tid}")
                await self.forecast_on_tournament(tid, return_exceptions=True)
            await self._compute_brier_scores()
            self.export_predictions_to_csv()
            self.export_cost_report()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    bot = UpskillBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model=DEFAULT_FORECASTER, temperature=0.12),
            "parser": GeneralLlm(model=PARSER_MODEL, temperature=0.0),
            "researcher_claude": GeneralLlm(model=DEFAULT_FORECASTER, temperature=0.12),
            "summarizer": GeneralLlm(model=SUMMARIZER_MODEL, temperature=0.0),
        },
    )

    tournament_ids = [32916, "ACX2026", "minibench", "market-pulse-26q1"]
    logger.info("🚀 Starting UpskillBot (Claude-only, calibrated prompts)...")
    asyncio.run(bot.run_all_tournaments(tournament_ids))
    logger.info("🏁 UpskillBot run completed successfully.")
