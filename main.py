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

# Pydantic for monkey-patching
from pydantic import model_validator

# Forecasting tools
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

# Rich for dashboard (optional)
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

# Tiktoken for cost tracking (optional)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger("UpskillBot")

# ------------------------------------------------------------------
# MODEL CONFIGURATION CONSTANTS
# ------------------------------------------------------------------
DEFAULT_FORECASTER = "openrouter/openai/gpt-5.1"
PARSER_MODEL = "openrouter/openai/gpt-4.1-mini"
RESEARCHER_GPT = "openrouter/openai/gpt-5"
RESEARCHER_CLAUDE = "openrouter/anthropic/claude-sonnet-4.5"
SUMMARIZER_MODEL = "openrouter/openai/gpt-4.1-mini"

MODEL_WEIGHTS = {
    RESEARCHER_GPT: 0.2,
    DEFAULT_FORECASTER: 0.5,
    RESEARCHER_CLAUDE: 0.3,
}

# ------------------------------------------------------------------
# MONKEY-PATCH: Fix PredictedOptionList validator
# ------------------------------------------------------------------
@model_validator(mode='after')
def _normalize_probs(self: PredictedOptionList):
    if not self.predicted_options:
        return self
    total = sum(p.probability for p in self.predicted_options)
    if total <= 0:
        logger.warning(f"PredictedOptionList sum is {total}. Cannot normalize. Raw: {self.predicted_options}")
        return self
    if abs(total - 1.0) > 0.001:
        logger.info(f"Normalizing probabilities (sum={total})")
        for opt in self.predicted_options:
            opt.probability = opt.probability / total
    for opt in self.predicted_options:
        opt.probability = max(0.0, min(1.0, opt.probability))
    return self

PredictedOptionList.__pydantic_post_validate__ = _normalize_probs
logger.info("✅ Monkey-patched PredictedOptionList validator.")


# -----------------------------
# Helper: Pure-Python median
# -----------------------------
def median(lst: List[float]) -> float:
    if not lst:
        raise ValueError("median() arg is an empty sequence")
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0
    else:
        return float(sorted_lst[mid])


# -----------------------------
# SAFE TAVILY QUERY BUILDER
# -----------------------------
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
        else:
            return q

    first_sent = q.split('.')[0].strip()
    if len(first_sent) > max_chars:
        return textwrap.shorten(first_sent, width=max_chars, placeholder="…")

    remaining = max_chars - len(first_sent) - 3
    if remaining > 10 and bg:
        bg_part = textwrap.shorten(bg, width=remaining, placeholder="…")
        combo = f"{first_sent} — {bg_part}"
        if len(combo) <= max_chars:
            return combo

    return textwrap.shorten(q, width=max_chars, placeholder="…")


# -----------------------------
# STRICT QUERY TRUNCATION
# -----------------------------
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


# -----------------------------
# UPSKILL BOT — FINAL CLEANED VERSION
# -----------------------------
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
        self._research_cache: Dict[str, str] = {}  # ✅ FIXED: was [] before!

        # Cost tracking
        self._cost_tracker = {}
        self._model_pricing = {
            "gpt-4.1-mini": {"input": 0.15, "output": 0.60},
            "gpt-5": {"input": 3.00, "output": 12.00},
            "gpt-5.1": {"input": 5.00, "output": 15.00},
            "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        }
        self._encoding_cache = {}

        # Dashboard
        self._console = Console() if RICH_AVAILABLE else None
        self._live_display = None
        self._questions_processed = 0
        self._questions_total = 0

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": DEFAULT_FORECASTER,
            "parser": PARSER_MODEL,
            "researcher_gpt": RESEARCHER_GPT,
            "researcher_claude": RESEARCHER_CLAUDE,
            "summarizer": SUMMARIZER_MODEL,
        }

    def _get_encoding(self, model_name: str):
        if not TIKTOKEN_AVAILABLE:
            return None
        if model_name in self._encoding_cache:
            return self._encoding_cache[model_name]
        if "gpt-4" in model_name or "gpt-5" in model_name:
            enc = tiktoken.get_encoding("cl100k_base")
        elif "claude" in model_name:
            enc = tiktoken.get_encoding("cl100k_base")
        else:
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

        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        return cost

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
        return await loop.run_in_executor(
            None,
            lambda: self.tavily.search(query=query.strip(), **kwargs),
        )

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
        """Safely extract median from NumericDistribution."""
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
            return normalized[0].value
        if normalized[0].percentile >= 0.5:
            return normalized[0].value
        if normalized[-1].percentile <= 0.5:
            return normalized[-1].value
        for i in range(len(normalized) - 1):
            p1, p2 = normalized[i], normalized[i + 1]
            if p1.percentile <= 0.5 <= p2.percentile:
                frac = (0.5 - p1.percentile) / (p2.percentile - p1.percentile)
                return p1.value + frac * (p2.value - p1.value)
        return normalized[-1].value

    def _interpolate_percentile(self, percentiles: List[Percentile], target_p: float) -> float:
        sorted_pts = sorted(percentiles, key=lambda x: x.percentile)
        if not sorted_pts:
            return 0.0
        if target_p <= sorted_pts[0].percentile:
            return sorted_pts[0].value
        if target_p >= sorted_pts[-1].percentile:
            return sorted_pts[-1].value
        for i in range(len(sorted_pts) - 1):
            p1, p2 = sorted_pts[i], sorted_pts[i + 1]
            if p1.percentile <= target_p <= p2.percentile:
                frac = (target_p - p1.percentile) / (p2.percentile - p1.percentile)
                return p1.value + frac * (p2.value - p1.value)
        return sorted_pts[-1].value

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
                    max_results=3,
                    days=180,
                )
                snippets = [
                    f"[{i+1}] {r['title']}: {textwrap.shorten(r['content'], width=150, placeholder='…')}"
                    for i, r in enumerate(recent.get("results", [])[:3])
                ]
                recent_summary = ("\n".join(snippets) if snippets else "[No recent results]")
            except Exception as e:
                logger.error(f"Recent Tavily failed: {e}")
                recent_summary = f"[Error: {e}]"

            historical_summary = "[Historical trends pending]"
            try:
                historical_query = strict_truncate_query(base_query, "What is the historical base rate or long-term trend?", 395)
                historical = await self._tavily_search_limited(
                    historical_query,
                    search_depth="advanced",
                    max_results=3,
                )
                snippets = [
                    f"[{i+1}] {r['title']}: {textwrap.shorten(r['content'], width=150, placeholder='…')}"
                    for i, r in enumerate(historical.get("results", [])[:3])
                ]
                historical_summary = ("\n".join(snippets) if snippets else "[No historical data]")
            except Exception as e:
                logger.error(f"Historical Tavily failed: {e}")
                historical_summary = f"[Error: {e}]"

            base_rate_prompt = clean_indents(f"""
                Estimate the historical base rate of events like this.
                Use only general knowledge or inferred trends.
                Output format: "Base rate: X%" or "Unknown".
            """)
            base_rate = await self._invoke_with_cost_tracking("researcher_gpt", base_rate_prompt)

            gpt_prompt = clean_indents(f"""
                You are a Good Judgment Project forecaster. Today: {datetime.now().strftime('%Y-%m-%d')}.
                Question: {question.question_text}
                Background: {question.background_info or 'None'}
                Be evidence-based, avoid narrative fallacy, anchor to base rates.
            """)
            gpt_response = await self._invoke_with_cost_tracking("researcher_gpt", gpt_prompt)

            claude_prompt = clean_indents(f"""
                Claude Sonnet 4.5. Analyze key uncertainties and structural drivers.
                Output only high-signal insights.
            """)
            claude_response = await self._invoke_with_cost_tracking("researcher_claude", claude_prompt)

            full_research = clean_indents(
                f"""
                ### UpskillBot Research (as of {today_str})
                --- BASE RATE ---
                {base_rate}

                --- RECENT DEVELOPMENTS (last 6mo) ---
                {recent_summary}

                --- HISTORICAL TRENDS ---
                {historical_summary}

                --- GPT-5 ANALYSIS ---
                {gpt_response}

                --- CLAUDE SONNET REVIEW ---
                {claude_response}
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
        today = datetime.now().strftime('%Y-%m-%d')
        decompose_instr = "Decompose into 3–5 key factors. Estimate each. Then synthesize."
        calib_instr = "You are calibrated: your 70% predictions resolve ~70% of the time. Avoid overconfidence."

        initial_prompt = clean_indents(f"""
            You are a superforecaster trained on the Good Judgment Project. Today: {today}.
            Question: {question.question_text}
            Research: {research}
            {decompose_instr}
            {calib_instr}
            Final line: "Probability: ZZ%"
        """)
        initial_reasoning = await self._invoke_with_cost_tracking("default", initial_prompt)

        red_team_prompt = clean_indents(f"""
            You are a skeptical expert who believes the above forecast is wrong.
            List 3 strongest counterarguments and evidence that would falsify it.
        """)
        red_team_response = await self._invoke_with_cost_tracking("researcher_claude", red_team_prompt)

        final_prompt = clean_indents(f"""
            Original reasoning:
            {initial_reasoning}

            Red team challenge:
            {red_team_response}

            Revise your forecast if warranted. Keep final line format.
        """)
        revised_reasoning = await self._invoke_with_cost_tracking("default", final_prompt)

        prob = 0.5
        try:
            if is_binary:
                pred: BinaryPrediction = await structure_output(
                    revised_reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
                )
                prob = max(0.01, min(0.99, pred.prediction_in_decimal))
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
        prompt = clean_indents(f"""
            {DEFAULT_FORECASTER.split('/')[-1]} forecaster. Options: {question.options}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Output format:
            Option_A: XX%
            Option_B: YY%
        """)
        reasoning = await self._invoke_with_cost_tracking("default", prompt)
        try:
            pred: PredictedOptionList = await structure_output(
                reasoning, PredictedOptionList, model=self.get_llm("parser", "llm"),
                additional_instructions=f"Options: {question.options}"
            )
        except Exception as e:
            logger.warning(f"MC parse fail Q{getattr(question, 'id', 'unknown')}: {e}")
            pred = PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=opt, probability=round(100.0 / len(question.options), 1))
                    for opt in question.options
                ]
            )

        prob_dict = {opt.option_name: opt.probability for opt in pred.predicted_options}
        top_opt = max(prob_dict, key=prob_dict.get)
        top_prob = prob_dict[top_opt] / 100.0
        self._record_prediction(question, top_prob, reasoning, extra={"top_option": top_opt})
        return ReasonedPrediction(prediction_value=pred, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        lower_msg = f"Cannot be lower than {question.lower_bound}." if not question.open_lower_bound else f"Unlikely below {question.lower_bound}."
        upper_msg = f"Cannot be higher than {question.upper_bound}." if not question.open_upper_bound else f"Unlikely above {question.upper_bound}."

        prompt = clean_indents(f"""
            GPT-5.1 quantitative forecaster.
            Question: {question.question_text}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            {lower_msg} {upper_msg}
            Decompose into key drivers. Output percentiles.
        """)
        reasoning = await self._invoke_with_cost_tracking("default", prompt)
        try:
            pct_list: list[Percentile] = await structure_output(
                reasoning, list[Percentile], model=self.get_llm("parser", "llm")
            )
            dist = NumericDistribution.from_question(pct_list, question)
        except Exception as e:
            logger.warning(f"Numeric parse fail: {e}")
            lo, hi = question.lower_bound, question.upper_bound
            fallback = [Percentile(p, lo + (hi - lo) * p / 100) for p in [10,20,40,60,80,90]]
            dist = NumericDistribution.from_question(fallback, question)

        median_val = self._get_numeric_median(dist)
        self._record_prediction(question, None, reasoning, extra={"median": median_val})
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        models = list(MODEL_WEIGHTS.keys())
        predictions = []
        reasonings = []
        model_names = []

        for model in models:
            original_default = self._llms.get("default")
            original_parser = self._llms.get("parser")
            self._llms["default"] = GeneralLlm(model=model)
            self._llms["parser"] = GeneralLlm(model=PARSER_MODEL)

            try:
                if isinstance(question, BinaryQuestion):
                    pred = await self._run_forecast_on_binary(question, research)
                elif isinstance(question, MultipleChoiceQuestion):
                    pred = await self._run_forecast_on_multiple_choice(question, research)
                elif isinstance(question, NumericQuestion):
                    pred = await self._run_forecast_on_numeric(question, research)
                else:
                    raise ValueError(f"Unsupported: {type(question)}")
                predictions.append(pred.prediction_value)
                reasonings.append(pred.reasoning)
                model_names.append(model)
            finally:
                self._llms["default"] = original_default
                self._llms["parser"] = original_parser

        if isinstance(question, BinaryQuestion):
            weighted_vals = []
            for pred, model in zip(predictions, model_names):
                weight = MODEL_WEIGHTS[model]
                count = max(1, int(weight * 100))
                weighted_vals.extend([pred] * count)
            median_val = median(weighted_vals)
            std_dev = (sum((p - median_val)**2 for p in predictions) / len(predictions)) ** 0.5
            self._record_prediction(question, median_val, " | ".join(reasonings), extra={
                "prediction_std": std_dev,
                "confidence_estimate": max(0.1, 1.0 - std_dev)
            })
            return ReasonedPrediction(prediction_value=median_val, reasoning=" | ".join(reasonings))

        elif isinstance(question, MultipleChoiceQuestion):
            options = question.options
            weighted_probs = {opt: [] for opt in options}
            for pred, model in zip(predictions, model_names):
                weight = MODEL_WEIGHTS[model]
                prob_map = {po.option_name: po.probability for po in pred.predicted_options}
                for opt in options:
                    weighted_probs[opt].append((prob_map.get(opt, 0.0), weight))
            avg_probs = {}
            for opt, vals in weighted_probs.items():
                total_weight = sum(w for _, w in vals)
                if total_weight == 0:
                    avg_probs[opt] = 1.0 / len(options)
                else:
                    avg_probs[opt] = sum(v * w for v, w in vals) / total_weight
            total = sum(avg_probs.values())
            if total > 0:
                avg_probs = {k: v / total for k, v in avg_probs.items()}
            pred_list = PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=k, probability=v) for k, v in avg_probs.items()
                ]
            )
            return ReasonedPrediction(prediction_value=pred_list, reasoning=" | ".join(reasonings))

        elif isinstance(question, NumericQuestion):
            target_pts = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            median_pcts = []
            for pt in target_pts:
                weighted_vals = []
                for dist, model in zip(predictions, model_names):
                    found = False
                    for item in dist.declared_percentiles:
                        if abs(item.percentile - pt) < 0.01:
                            weighted_vals.extend([item.value] * int(MODEL_WEIGHTS[model] * 100))
                            found = True
                            break
                    if not found:
                        interp_val = self._interpolate_percentile(dist.declared_percentiles, pt)
                        weighted_vals.extend([interp_val] * int(MODEL_WEIGHTS[model] * 100))
                median_pcts.append(Percentile(percentile=pt, value=median(weighted_vals)))
            final_dist = NumericDistribution.from_question(median_pcts, question)
            return ReasonedPrediction(prediction_value=final_dist, reasoning=" | ".join(reasonings))

        else:
            return ReasonedPrediction(prediction_value=predictions[0], reasoning=" | ".join(reasonings))

    async def _compute_brier_scores(self):
        try:
            # Use MetaculusClient without auth (public questions only)
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
                    pred = rec["predicted_prob"]
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
                logger.info(f"📊 Avg Brier (n={scored}): {brier_sum / scored:.4f}")
                logger.info(f"📊 Avg Log Score (n={scored}): {log_score_sum / scored:.4f}")
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


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    bot = UpskillBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model=DEFAULT_FORECASTER, temperature=0.2),
            "parser": GeneralLlm(model=PARSER_MODEL, temperature=0.0),
            "researcher_gpt": GeneralLlm(model=RESEARCHER_GPT, temperature=0.3),
            "researcher_claude": GeneralLlm(model=RESEARCHER_CLAUDE, temperature=0.3),
            "summarizer": GeneralLlm(model=SUMMARIZER_MODEL, temperature=0.0),
        },
    )

    tournament_ids = [32916, "ACX2026", "minibench", "market-pulse-26q1"]
    logger.info("🚀 Starting UpskillBot (Final Clean Version)...")
    asyncio.run(bot.run_all_tournaments(tournament_ids))
    logger.info("🏁 UpskillBot run completed successfully.")
