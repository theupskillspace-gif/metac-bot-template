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
import httpx
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

logger = logging.getLogger("UpskillBot")

DEFAULT_FORECASTER = "openrouter/anthropic/claude-sonnet-4.5"
PARSER_MODEL = "openrouter/openai/gpt-4.1-mini"
SUMMARIZER_MODEL = "openrouter/openai/gpt-4.1-mini"


# =========================================================
# 🔍 EXA SEARCHER
# =========================================================
class ExaSearcher:
    """
    Lightweight async client for Exa search.
    Requires EXA_API_KEY env var; if missing, Exa search is skipped gracefully.
    """

    def __init__(self):
        self.api_key = os.getenv("EXA_API_KEY")
        self.base_url = "https://api.exa.ai/search"
        if not self.api_key:
            logger.warning("EXA_API_KEY not set; Exa search disabled.")

    async def search(self, query: str, num_results: int = 4) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        payload = {
            "query": query,
            "numResults": num_results,
            "type": "neural",
            "useAutoprompt": True,
            "category": "news",
        }
        headers = {"Content-Type": "application/json", "x-api-key": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(self.base_url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                return data.get("results", []) or []
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return []


# =========================================================
# ✅ Normalize multiple choice probabilities (robust)
# =========================================================
@model_validator(mode="after")
def _normalize_probs(self: PredictedOptionList):
    if not getattr(self, "predicted_options", None):
        return self

    probs = [float(p.probability) for p in self.predicted_options]
    total = sum(probs)

    if total <= 0:
        logger.warning(f"PredictedOptionList sum is {total}. Raw: {self.predicted_options}")
        return self

    # If model returned percentages (e.g., 45, 30, 25)
    if total > 1.5:
        for opt in self.predicted_options:
            opt.probability = float(opt.probability) / 100.0
        probs = [float(p.probability) for p in self.predicted_options]
        total = sum(probs)
        if total <= 0:
            return self

    # Normalize to sum to 1
    if abs(total - 1.0) > 0.001:
        for opt in self.predicted_options:
            opt.probability = float(opt.probability) / total

    # Clamp and renormalize
    for opt in self.predicted_options:
        opt.probability = max(0.0, min(1.0, float(opt.probability)))
    total2 = sum(float(p.probability) for p in self.predicted_options)
    if total2 > 0 and abs(total2 - 1.0) > 1e-6:
        for opt in self.predicted_options:
            opt.probability = float(opt.probability) / total2

    return self


PredictedOptionList.__pydantic_post_validate__ = _normalize_probs


# =========================================================
# 🧰 Helpers
# =========================================================
def build_tavily_query(question: MetaculusQuestion, max_chars: int = 300) -> str:
    q = (question.question_text or "").strip()
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


def clamp01(p: float) -> float:
    return max(0.0, min(1.0, p))


def conservative_shrink(p: float, strength: float = 0.18) -> float:
    """
    Shrink probabilities slightly toward 0.5 to be conservative unless evidence is strong.
    strength=0.18 means move 18% of the way from p to 0.5.
    """
    p = clamp01(p)
    return (1.0 - strength) * p + strength * 0.5


def conservative_clip(p: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return max(lo, min(hi, p))


def extract_probability_percent(text: str) -> Optional[float]:
    """
    Extracts 'Probability: ZZ%' from free text. Returns decimal in [0,1] or None.
    """
    m = re.search(r"Probability\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?", text, flags=re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    if val > 1.0:
        return clamp01(val / 100.0)
    return clamp01(val)


def shorten_reasoning(text: str, max_chars: int = 900) -> str:
    """
    Keep reasoning short for posting: strip excess whitespace and truncate.
    """
    t = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


# =========================================================
# 🤖 UpskillBot (conservative, no rich, Tavily + Exa research)
# =========================================================
class UpskillBot(ForecastBot):
    """
    Conservative forecasting bot:
      - Uses Tavily + Exa for research (if keys present).
      - Keeps written explanations short.
      - Applies gentle shrink toward 50% and clips away from 0/1.
      - No Rich UI/dashboard.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            raise ValueError("TAVILY_API_KEY must be set.")
        self.tavily = TavilyClient(api_key=tavily_key)

        self.exa = ExaSearcher()

        self._prediction_records: List[Dict[str, Any]] = []
        self._research_cache: Dict[str, str] = {}

        # Optional cost tracking (only active if tiktoken installed)
        self._cost_tracker: Dict[str, Dict[str, int]] = {}
        self._model_pricing = {
            "gpt-4.1-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        }
        self._encoding_cache: Dict[str, Any] = {}

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": DEFAULT_FORECASTER,
            "parser": PARSER_MODEL,
            "summarizer": SUMMARIZER_MODEL,
        }

    def _get_encoding(self, model_name: str):
        try:
            import tiktoken  # local import to avoid hard dependency
        except ImportError:
            return None
        if model_name in self._encoding_cache:
            return self._encoding_cache[model_name]
        enc = tiktoken.get_encoding("cl100k_base")
        self._encoding_cache[model_name] = enc
        return enc

    def _estimate_cost(self, model_path: str, prompt: str, completion: str) -> float:
        enc = self._get_encoding(model_path.split("/")[-1])
        if not enc:
            return 0.0

        if model_path not in self._cost_tracker:
            self._cost_tracker[model_path] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

        model_key = model_path.split("/")[-1]
        pricing = self._model_pricing.get(model_key, {"input": 1.0, "output": 3.0})

        input_tokens = len(enc.encode(prompt))
        output_tokens = len(enc.encode(completion))

        self._cost_tracker[model_path]["input_tokens"] += input_tokens
        self._cost_tracker[model_path]["output_tokens"] += output_tokens
        self._cost_tracker[model_path]["calls"] += 1

        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    async def _invoke(self, model_name: str, prompt: str) -> str:
        llm = self.get_llm(model_name, "llm")
        response = await llm.invoke(prompt)
        self._estimate_cost(llm.model, prompt, response)
        return response

    async def _tavily_search(self, query: str, **kwargs) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.tavily.search(query=query.strip(), **kwargs))

    def _is_stock_question(self, question: MetaculusQuestion) -> bool:
        text = " ".join(
            [question.question_text or "", question.background_info or "", question.resolution_criteria or ""]
        ).lower()
        patterns = [
            r"\b(?:stock|equity|share|s&p|nasdaq|dow|ticker)\b",
            r"\b\$\s*[a-z]{1,5}\b",
        ]
        return any(re.search(pat, text) for pat in patterns)

    def _estimate_question_difficulty(self, question: MetaculusQuestion) -> float:
        text = ((question.question_text or "") + " " + (question.background_info or "")).lower()
        now = datetime.now(timezone.utc)
        days_to_close = (
            (question.close_time - now).total_seconds() / 86400 if getattr(question, "close_time", None) else 365
        )
        base_rate_hint = any(w in text for w in ["rare", "unlikely", "first time", "never before", "unprecedented"])
        long_horizon = days_to_close > 365
        vague_resolution = "ambiguous" in (question.resolution_criteria or "").lower()
        return min(1.0, 0.3 + 0.3 * long_horizon + 0.2 * base_rate_hint + 0.2 * vague_resolution)

    async def run_research(self, question: MetaculusQuestion) -> str:
        qid = getattr(question, "id", getattr(question, "question_id", hash(question.question_text or "")))
        cache_key = str(qid)
        if cache_key in self._research_cache:
            return self._research_cache[cache_key]

        async with self._concurrency_limiter:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            base_query = build_tavily_query(question)

            # Tavily recent
            try:
                recent_query = strict_truncate_query(base_query, "Developments in last 6 months.", 395)
                recent = await self._tavily_search(
                    recent_query,
                    search_depth="advanced",
                    max_results=4,
                    days=180,
                )
                recent_snips = [
                    f"[T{i+1}] {r.get('title','')}: {textwrap.shorten((r.get('content') or ''), width=160, placeholder='…')}"
                    for i, r in enumerate(recent.get("results", [])[:4])
                ]
                recent_summary = "\n".join(recent_snips) if recent_snips else "[T] No recent results"
            except Exception as e:
                logger.error(f"Recent Tavily failed: {e}")
                recent_summary = f"[T] Error: {e}"

            # Tavily historical/base rate
            try:
                historical_query = strict_truncate_query(base_query, "Historical base rates / reference class.", 395)
                historical = await self._tavily_search(
                    historical_query,
                    search_depth="advanced",
                    max_results=4,
                )
                hist_snips = [
                    f"[H{i+1}] {r.get('title','')}: {textwrap.shorten((r.get('content') or ''), width=160, placeholder='…')}"
                    for i, r in enumerate(historical.get("results", [])[:4])
                ]
                historical_summary = "\n".join(hist_snips) if hist_snips else "[H] No historical results"
            except Exception as e:
                logger.error(f"Historical Tavily failed: {e}")
                historical_summary = f"[H] Error: {e}"

            # Exa
            try:
                exa_query = strict_truncate_query(base_query, "", 395)
                exa_results = await self.exa.search(exa_query, num_results=4) if self.exa else []
                if exa_results:
                    exa_snips = [
                        f"[E{i+1}] {r.get('title','')}: {textwrap.shorten((r.get('text') or r.get('snippet') or ''), width=160, placeholder='…')}"
                        for i, r in enumerate(exa_results[:4])
                    ]
                    exa_summary = "\n".join(exa_snips)
                else:
                    exa_summary = "[E] No Exa results (or EXA_API_KEY missing)"
            except Exception as e:
                logger.error(f"Exa search failed: {e}")
                exa_summary = f"[E] Error: {e}"

            research = clean_indents(
                f"""
                ### Research (as of {today_str})
                {recent_summary}

                {historical_summary}

                {exa_summary}
                """
            ).strip()

            self._research_cache[cache_key] = research
            return research

    def _record_prediction(
        self,
        question: MetaculusQuestion,
        prob: Optional[float],
        reasoning: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            qid = getattr(question, "id", None) or getattr(
                question, "question_id", f"anon_{hash(question.question_text or '') % 10000}"
            )
            record: Dict[str, Any] = {
                "question_id": qid,
                "page_url": getattr(question, "page_url", "N/A"),
                "title": (getattr(question, "question_text", "Unknown") or "")[:120],
                "type": question.__class__.__name__,
                "predicted_prob": prob,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "difficulty_score": self._estimate_question_difficulty(question),
                "is_stock": self._is_stock_question(question),
                "reasoning_snippet": shorten_reasoning(reasoning, 400).replace("\n", " "),
            }
            if extra:
                for k, v in extra.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        record[k] = v
                    else:
                        try:
                            record[k] = json.dumps(v, ensure_ascii=False)
                        except Exception:
                            record[k] = str(v)
            self._prediction_records.append(record)
        except Exception as e:
            logger.debug(f"Non-fatal: prediction record skipped ({e})")

    async def _run_forecast_with_red_team(
        self, question: MetaculusQuestion, research: str
    ) -> Tuple[str, float]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        initial_prompt = clean_indents(
            f"""
            You are a calibrated, conservative superforecaster. Today (UTC): {today}.
            Rules:
            - Prefer outside view first (reference class/base rate).
            - Be conservative: when uncertain, move toward 50%.
            - Avoid unjustified extremes (rarely below 5% or above 95% unless strong evidence).
            - Keep the explanation short: 6–10 bullets max.
            - End with EXACT line: Probability: ZZ%

            QUESTION:
            {question.question_text}

            BACKGROUND:
            {question.background_info or 'None'}

            RESOLUTION CRITERIA:
            {question.resolution_criteria or 'None'}

            RESEARCH:
            {research}

            OUTPUT FORMAT:
            Approach: <1 sentence>
            Evidence: <6-10 bullets, each ≤ 1 line>
            Uncertainties: <2 bullets>
            Final: <1 sentence>
            Probability: ZZ%
            """
        )
        initial_text = await self._invoke("default", initial_prompt)

        red_team_prompt = clean_indents(
            f"""
            You are a skeptical reviewer. Critique the forecast and push back against overconfidence.
            Provide:
            - 3 strongest counterarguments (bullets)
            - 2 signposts that would change the forecast (bullets)
            - A calibration note: should this be closer to 50%?

            FORECAST:
            {initial_text}

            QUESTION:
            {question.question_text}

            RESEARCH:
            {research}
            """
        )
        critique = await self._invoke("default", red_team_prompt)

        final_prompt = clean_indents(
            f"""
            Revise conservatively based on the critique.
            Rules:
            - Make the smallest necessary adjustment.
            - If evidence is mixed/weak, shrink toward 50%.
            - Keep output short, same format, and end with: Probability: ZZ%

            ORIGINAL:
            {initial_text}

            CRITIQUE:
            {critique}
            """
        )
        revised_text = await self._invoke("default", final_prompt)

        prob = None
        try:
            parsed: BinaryPrediction = await structure_output(
                revised_text, BinaryPrediction, model=self.get_llm("parser", "llm")
            )
            prob = float(parsed.prediction_in_decimal)
        except Exception:
            prob = extract_probability_percent(revised_text)

        if prob is None:
            prob = 0.5

        prob = conservative_shrink(prob, strength=0.18)
        prob = conservative_clip(prob, lo=0.02, hi=0.98)

        # Force final displayed probability to match parsed number
        revised_text = re.sub(
            r"Probability\s*:\s*[0-9]+(?:\.[0-9]+)?\s*%?",
            f"Probability: {prob*100:.1f}%",
            revised_text,
            flags=re.IGNORECASE,
        )

        return shorten_reasoning(revised_text, 900), prob

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        reasoning, prob = await self._run_forecast_with_red_team(question, research)
        self._record_prediction(question, prob, reasoning)
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        prompt = clean_indents(
            f"""
            You are a calibrated, conservative forecaster. Today (UTC): {today}.
            Rules:
            - Prefer outside view first.
            - Be conservative: avoid extreme allocations unless strong evidence.
            - Probabilities must be decimals in [0,1] and sum to 1.
            - Keep reasoning short (≤ 8 bullets).

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
            Approach: <1 sentence>
            Evidence: <≤8 bullets>
            Then list options as:
            <Option text>: 0.23
            """
        )
        reasoning = await self._invoke("default", prompt)

        try:
            pred: PredictedOptionList = await structure_output(
                reasoning,
                PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=f"Options: {question.options}. Probabilities must be decimals (0-1) summing to 1.",
            )
        except Exception as e:
            logger.warning(f"MC parse failed Q{getattr(question, 'id', 'unknown')}: {e}")
            p = 1.0 / max(1, len(question.options))
            pred = PredictedOptionList(
                predicted_options=[PredictedOption(option_name=opt, probability=p) for opt in question.options]
            )

        # Conservative smoothing: mix with uniform to avoid overconfident spikes
        uniform = 1.0 / max(1, len(question.options))
        smoothed = []
        for opt in pred.predicted_options:
            p = float(opt.probability)
            p = (0.85 * p) + (0.15 * uniform)
            smoothed.append(PredictedOption(option_name=opt.option_name, probability=p))
        pred = PredictedOptionList(predicted_options=smoothed)

        prob_dict = {opt.option_name: float(opt.probability) for opt in pred.predicted_options}
        top_opt = max(prob_dict, key=prob_dict.get) if prob_dict else "N/A"
        top_prob = float(prob_dict[top_opt]) if prob_dict else None

        reasoning_short = shorten_reasoning(reasoning, 900)
        self._record_prediction(question, top_prob, reasoning_short, extra={"top_option": top_opt})
        return ReasonedPrediction(prediction_value=pred, reasoning=reasoning_short)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lower_msg = (
            f"Lower bound: {question.lower_bound}." if question.lower_bound is not None else "No explicit lower bound."
        )
        upper_msg = (
            f"Upper bound: {question.upper_bound}." if question.upper_bound is not None else "No explicit upper bound."
        )

        prompt = clean_indents(
            f"""
            You are a calibrated, conservative quantitative forecaster. Today (UTC): {today}.
            Rules:
            - Start with outside view / reference class.
            - Be conservative: widen tails if uncertain.
            - Respect bounds if given.
            - Keep reasoning short (≤ 8 bullets).
            - Output MUST include a parsable percentile list.

            QUESTION:
            {question.question_text}

            BACKGROUND:
            {question.background_info or 'None'}

            RESOLUTION CRITERIA:
            {question.resolution_criteria or 'None'}

            BOUNDS:
            {lower_msg} {upper_msg}

            UNITS:
            {question.unit_of_measure or "Unknown"}

            RESEARCH:
            {research}

            OUTPUT:
            Approach: <1 sentence>
            Evidence: <≤8 bullets>
            Then a JSON-like list:
            [
              {{"percentile":0.1,"value":123}},
              {{"percentile":0.2,"value":...}},
              {{"percentile":0.4,"value":...}},
              {{"percentile":0.6,"value":...}},
              {{"percentile":0.8,"value":...}},
              {{"percentile":0.9,"value":...}}
            ]
            """
        )
        reasoning = await self._invoke("default", prompt)

        try:
            pct_list: list[Percentile] = await structure_output(
                reasoning, list[Percentile], model=self.get_llm("parser", "llm")
            )
            dist = NumericDistribution.from_question(pct_list, question)
        except Exception as e:
            logger.warning(f"Numeric parse failed: {e}")
            lo = float(question.lower_bound if question.lower_bound is not None else 0.0)
            hi = float(question.upper_bound if question.upper_bound is not None else lo + 1.0)
            fallback_ps = [0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
            fallback = [Percentile(percentile=p, value=lo + (hi - lo) * p) for p in fallback_ps]
            dist = NumericDistribution.from_question(fallback, question)

        reasoning_short = shorten_reasoning(reasoning, 900)
        self._record_prediction(question, None, reasoning_short)
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning_short)

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
                r for r in self._prediction_records if r["type"] == "BinaryQuestion" and r["predicted_prob"] is not None
            ]
            question_ids = [
                r["question_id"]
                for r in binary_records
                if isinstance(r["question_id"], (int, str)) and r["question_id"] not in ("N/A", "unknown")
            ]
            if not question_ids:
                return

            all_qs = await client.get_questions_by_ids(question_ids)
            resolved_qs = [q for q in all_qs if isinstance(q, BinaryQuestion) and q.resolution in ("yes", "no")]

            brier_sum = log_score_sum = scored = 0.0
            for q in resolved_qs:
                rec = next((r for r in binary_records if r["question_id"] == q.id), None)
                if not rec:
                    continue
                pred = float(rec["predicted_prob"])
                actual = 1.0 if q.resolution == "yes" else 0.0

                brier = (pred - actual) ** 2
                eps = 1e-6
                clipped_pred = max(eps, min(1 - eps, pred))
                log_score = actual * math.log(clipped_pred) + (1 - actual) * math.log(1 - clipped_pred)

                brier_sum += brier
                log_score_sum += log_score
                scored += 1

                rec.update(
                    {
                        "resolution": q.resolution,
                        "actual": actual,
                        "brier_score": round(brier, 4),
                        "log_score": round(log_score, 4),
                    }
                )

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
            records.append(
                {
                    "model": model,
                    "calls": stats["calls"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "estimated_cost_usd": round(cost, 6),
                }
            )
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Exported cost report to {filepath}")
        logger.info(f"💰 Total estimated cost: ${total_cost:.4f}")

    async def run_all_tournaments(self, tournament_ids: List):
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
            "default": GeneralLlm(model=DEFAULT_FORECASTER, temperature=0.10),
            "parser": GeneralLlm(model=PARSER_MODEL, temperature=0.0),
            "summarizer": GeneralLlm(model=SUMMARIZER_MODEL, temperature=0.0),
        },
    )

    tournament_ids = [32916, "ACX2026", "minibench", "market-pulse-26q1"]
    logger.info("🚀 Starting UpskillBot (conservative, Tavily+Exa, short writeups)...")
    asyncio.run(bot.run_all_tournaments(tournament_ids))
    logger.info("🏁 UpskillBot run completed.")
