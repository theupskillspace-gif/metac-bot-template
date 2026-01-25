from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import textwrap
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

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
        MetaculusApi,
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

logger = logging.getLogger("UpskillBot")

# ------------------------------------------------------------------
# MODEL CONFIGURATION CONSTANTS
# ------------------------------------------------------------------
DEFAULT_FORECASTER = "openrouter/openai/gpt-5.1"
PARSER_MODEL = "openrouter/openai/gpt-4.1-mini"
RESEARCHER_GPT = "openrouter/openai/gpt-5"
RESEARCHER_CLAUDE = "openrouter/anthropic/claude-sonnet-4.5"
SUMMARIZER_MODEL = "openrouter/openai/gpt-4.1-mini"

# Model weights for weighted aggregation (higher = more trusted)
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
# TAVILY QUERY BUILDER
# -----------------------------
def build_tavily_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
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
# UPSKILL BOT — SUPERFORECASTER EDITION
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
        self._research_cache: Dict[str, str] = {}

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": DEFAULT_FORECASTER,
            "parser": PARSER_MODEL,
            "researcher_gpt": RESEARCHER_GPT,
            "researcher_claude": RESEARCHER_CLAUDE,
            "summarizer": SUMMARIZER_MODEL,
        }

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
        """Estimate epistemic difficulty (0–1)."""
        text = (question.question_text + " " + (question.background_info or "")).lower()
        now = datetime.now(timezone.utc)
        days_to_close = (question.close_time - now).total_seconds() / 86400 if question.close_time else 365
        base_rate_hint = any(w in text for w in ["rare", "unlikely", "first time", "never before", "unprecedented"])
        long_horizon = days_to_close > 365
        vague_resolution = "ambiguous" in (question.resolution_criteria or "").lower()
        return min(1.0, 0.3 + 0.3 * long_horizon + 0.2 * base_rate_hint + 0.2 * vague_resolution)

    async def run_research(self, question: MetaculusQuestion) -> str:
        qid = getattr(question, "id", getattr(question, "question_id", hash(question.question_text)))
        cache_key = str(qid)
        if cache_key in self._research_cache:
            return self._research_cache[cache_key]

        async with self._concurrency_limiter:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            base_query = build_tavily_query(question)

            # Two-pronged search: recent + historical
            recent_summary = "[Recent developments pending]"
            historical_summary = "[Historical trends pending]"

            try:
                recent = await self._tavily_search_limited(
                    base_query + " Focus on developments in the last 6 months.",
                    search_depth="advanced",
                    max_results=3,
                    days=180,
                )
                recent_snippets = [
                    f"[{i+1}] {r['title']}: {textwrap.shorten(r['content'], width=150, placeholder='…')}"
                    for i, r in enumerate(recent.get("results", [])[:3])
                ]
                recent_summary = ("\n".join(recent_snippets) if recent_snippets else "[No recent results]")
            except Exception as e:
                logger.error(f"Recent Tavily failed: {e}")
                recent_summary = f"[Error: {e}]"

            try:
                historical = await self._tavily_search_limited(
                    base_query + " What is the historical base rate or long-term trend?",
                    search_depth="advanced",
                    max_results=3,
                )
                hist_snippets = [
                    f"[{i+1}] {r['title']}: {textwrap.shorten(r['content'], width=150, placeholder='…')}"
                    for i, r in enumerate(historical.get("results", [])[:3])
                ]
                historical_summary = ("\n".join(hist_snippets) if hist_snippets else "[No historical data]")
            except Exception as e:
                logger.error(f"Historical Tavily failed: {e}")
                historical_summary = f"[Error: {e}]"

            # Base rate estimation via LLM
            base_rate_prompt = clean_indents(f"""
                Estimate the historical base rate of events like this.
                Use only general knowledge or inferred trends.
                Output format: "Base rate: X%" or "Unknown".
            """)
            base_rate = await self.get_llm("researcher_gpt", "llm").invoke(base_rate_prompt)

            gpt_prompt = clean_indents(f"""
                You are a Good Judgment Project forecaster. Today: {datetime.now().strftime('%Y-%m-%d')}.
                Question: {question.question_text}
                Background: {question.background_info or 'None'}
                Be evidence-based, avoid narrative fallacy, anchor to base rates.
            """)
            claude_prompt = clean_indents(f"""
                Claude Sonnet 4.5. Analyze key uncertainties and structural drivers.
                Output only high-signal insights.
            """)

            gpt_response = await self.get_llm("researcher_gpt", "llm").invoke(gpt_prompt)
            claude_response = await self.get_llm("researcher_claude", "llm").invoke(claude_prompt)

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

    async def _run_forecast_with_red_team(
        self, question: MetaculusQuestion, research: str, is_binary: bool = True
    ) -> Tuple[str, float]:
        # Step 1: Initial forecast with decomposition
        today = datetime.now().strftime('%Y-%m-%d')
        decompose_instr = (
            "Decompose into 3–5 key factors. Estimate each. Then synthesize."
            if not isinstance(question, MultipleChoiceQuestion)
            else ""
        )
        calib_instr = (
            "You are calibrated: your 70% predictions resolve ~70% of the time. "
            "Avoid overconfidence. Anchor to base rates."
        )

        initial_prompt = clean_indents(f"""
            You are a superforecaster trained on the Good Judgment Project. Today: {today}.
            Question: {question.question_text}
            Research: {research}
            {decompose_instr}
            {calib_instr}
            Final line: "Probability: ZZ%" (binary) or option probabilities (MCQ).
        """)
        initial_reasoning = await self.get_llm("default", "llm").invoke(initial_prompt)

        # Step 2: Red team challenge
        red_team_prompt = clean_indents(f"""
            You are a skeptical expert who believes the above forecast is wrong.
            List 3 strongest counterarguments and evidence that would falsify it.
        """)
        red_team_response = await self.get_llm("researcher_claude", "llm").invoke(red_team_prompt)

        # Step 3: Revise
        final_prompt = clean_indents(f"""
            Original reasoning:
            {initial_reasoning}

            Red team challenge:
            {red_team_response}

            Revise your forecast if warranted. Keep final line format.
        """)
        revised_reasoning = await self.get_llm("default", "llm").invoke(final_prompt)

        # Extract probability
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
        # For MCQ, skip red team for simplicity (or extend if needed)
        prompt = clean_indents(f"""
            {DEFAULT_FORECASTER.split('/')[-1]} forecaster. Options: {question.options}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Output format:
            Option_A: XX%
            Option_B: YY%
        """)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
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
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
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

        median_val = dist.get_percentile_value(50)
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

        # Weighted aggregation
        if isinstance(question, BinaryQuestion):
            # Create weighted list for median
            weighted_vals = []
            for pred, model in zip(predictions, model_names):
                weight = MODEL_WEIGHTS[model]
                count = max(1, int(weight * 100))
                weighted_vals.extend([pred] * count)
            median_val = median(weighted_vals)
            # Compute std dev for confidence
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
            api = MetaculusApi()
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
            all_qs = await api.get_questions_by_ids(question_ids)
            resolved_qs = [q for q in all_qs if isinstance(q, BinaryQuestion) and q.resolution in ("yes", "no")]
            brier_sum = log_score_sum = scored = 0.0
            for q in resolved_qs:
                rec = next((r for r in binary_records if r["question_id"] == q.id), None)
                if rec:
                    pred = rec["predicted_prob"]
                    actual = 1.0 if q.resolution == "yes" else 0.0
                    brier = (pred - actual) ** 2
                    # Log score (proper scoring rule)
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

    async def run_all_tournaments(self, tournament_ids: List):
        for tid in tournament_ids:
            logger.info(f"▶ Forecasting tournament: {tid}")
            await self.forecast_on_tournament(tid, return_exceptions=True)
        await self._compute_brier_scores()
        self.export_predictions_to_csv()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    if os.getenv("METACULUS_TOKEN") or (os.getenv("METACULUS_EMAIL") and os.getenv("METACULUS_PASSWORD")):
        try:
            api = MetaculusApi()
            if os.getenv("METACULUS_TOKEN"):
                api.login_with_token(os.getenv("METACULUS_TOKEN"))
            else:
                api.login(os.getenv("METACULUS_EMAIL"), os.getenv("METACULUS_PASSWORD"))
            logger.info("✅ Metaculus auth successful.")
        except Exception as e:
            logger.error(f"Metaculus login failed: {e}")

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

    tournament_ids = [32916, "ACX2026", "minibench"]
    logger.info("🚀 Starting UpskillBot (Superforecaster Edition)...")
    asyncio.run(bot.run_all_tournaments(tournament_ids))
    logger.info("🏁 UpskillBot run completed successfully.")
