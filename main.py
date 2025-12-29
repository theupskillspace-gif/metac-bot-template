from __future__ import annotations

import argparse
import asyncio
import logging
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
        opt.probability = max(0.0, opt.probability)
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
# UPSKILL BOT — FINAL FIXED VERSION
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

    # ✅ FIXED: Added 'summarizer' (required by ForecastBot base class)
    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.1",        # Main forecaster
            "parser": "openrouter/openai/gpt-4.1-mini",    # Structured output
            "researcher_gpt": "openrouter/openai/gpt-5",   # Research
            "researcher_claude": "openrouter/anthropic/claude-sonnet-4.5",  # Research
            "summarizer": "openrouter/openai/gpt-4.1-mini",  # 🔑 CRITICAL: For research summarization
        }

    async def _tavily_search_limited(self, query: str) -> dict:
        async with self._tavily_lock:
            if self._tavily_query_count >= self._max_tavily_queries:
                raise RuntimeError(f"UpskillBot: Tavily limit ({self._max_tavily_queries}) reached.")
            self._tavily_query_count += 1
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.tavily.search(
                query=query.strip(),
                search_depth="advanced",
                include_answer=True,
                max_results=5,
                days=180,
            ),
        )

    def _is_stock_question(self, question: MetaculusQuestion) -> bool:
        text = " ".join([question.question_text, question.background_info or "", question.resolution_criteria or ""]).lower()
        patterns = [r"\b(?:stock|equity|share|s&p|nasdaq|dow|ticker)\b", r"\b\$?[a-z]{1,5}\b"]
        return any(re.search(pat, text) for pat in patterns)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            query = build_tavily_query(question)
            logger.debug(f"Tavily query ({len(query)}c): {repr(query)}")

            tavily_summary = "[Tavily pending]"
            try:
                response = await self._tavily_search_limited(query)
                answer = response.get("answer", "No answer.")
                results = response.get("results", [])
                snippets = [
                    f"[{i+1}] {r['title']}: {textwrap.shorten(r['content'], width=180, placeholder='…')}"
                    for i, r in enumerate(results[:5])
                ]
                tavily_summary = f"Answer: {answer}\n" + ("\n".join(snippets) if snippets else "[No results]")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Tavily failed Q{getattr(question, 'id', 'unknown')}: {error_msg}")
                if "400 characters" in error_msg:
                    try:
                        short_query = textwrap.shorten(query, width=200, placeholder="…")
                        response = self.tavily.search(query=short_query, search_depth="basic", max_results=3)
                        snippets = [f"[{i+1}] {r['title']}" for i, r in enumerate(response.get("results", []))]
                        tavily_summary = "[FALLBACK] " + ("\n".join(snippets) if snippets else "[No results]")
                    except Exception as e2:
                        tavily_summary = f"[Tavily failed: {error_msg} → {e2}]"
                else:
                    tavily_summary = f"[Tavily error: {error_msg}]"

            # Research supplements
            gpt_prompt = clean_indents(f"""
                Superforecaster. Today: {datetime.now().strftime('%Y-%m-%d')}.
                Question: {question.question_text}
                Background: {question.background_info or 'None'}
                Be sharp, evidence-based, no fluff.
            """)
            claude_prompt = clean_indents(f"""
                Claude Sonnet 4.5. Today: {datetime.now().strftime('%Y-%m-%d')}.
                Question: {question.question_text}
                Output only high-signal insights.
            """)

            gpt_response = await self.get_llm("researcher_gpt", "llm").invoke(gpt_prompt)
            claude_response = await self.get_llm("researcher_claude", "llm").invoke(claude_prompt)

            return clean_indents(
                f"""
                ### UpskillBot Research (as of {today_str})
                --- TAVILY ---
                {tavily_summary}

                --- GPT-5 ANALYSIS ---
                {gpt_response}

                --- CLAUDE SONNET REVIEW ---
                {claude_response}
                """
            )

    # ✅ FIXED: Safe _record_prediction (no more AttributeError)
    def _record_prediction(
        self,
        question: MetaculusQuestion,
        prob: Optional[float],
        reasoning: str,
        extra: Optional[Dict] = None,
    ):
        try:
            # Ultra-safe ID extraction
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
                "reasoning_snippet": reasoning[:500].replace("\n", " "),
            }
            if extra:
                record.update(extra)
            self._prediction_records.append(record)
        except Exception as e:
            logger.debug(f"Non-fatal: Skipped recording prediction ({e})")

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(f"""
            You are a GPT-5.1 superforecaster. Today: {datetime.now().strftime('%Y-%m-%d')}.
            Question: {question.question_text}
            Research: {research}
            → Be boldly calibrated. Final line: "Probability: ZZ%"
        """)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        try:
            pred: BinaryPrediction = await structure_output(
                reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
            )
            prob = max(0.01, min(0.99, pred.prediction_in_decimal))
        except Exception as e:
            logger.warning(f"Binary parse fail Q{getattr(question, 'id', 'unknown')}: {e}")
            prob = 0.5

        # ✅ Safe call
        self._record_prediction(question, prob, reasoning)
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(f"""
            GPT-5.1 forecaster. Options: {question.options}
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

        # ✅ Safe call
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
            Output:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        try:
            pct_list: list[Percentile] = await structure_output(
                reasoning, list[Percentile], model=self.get_llm("parser", "llm")
            )
            dist = NumericDistribution.from_question(pct_list, question)
        except Exception as e:
            logger.warning(f"Numeric parse fail Q{getattr(question, 'id', 'unknown')}: {e}")
            lo, hi = question.lower_bound, question.upper_bound
            fallback = [Percentile(p, lo + (hi - lo) * p / 100) for p in [10,20,40,60,80,90]]
            dist = NumericDistribution.from_question(fallback, question)

        median_val = dist.get_percentile_value(50)

        # ✅ Safe call
        self._record_prediction(question, None, reasoning, extra={"median": median_val})
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    # -----------------------------
    # COMMITTEE: GPT-5, GPT-5.1, CLAUDE SONNET 4.5
    # -----------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5.1",
            "openrouter/anthropic/claude-sonnet-4.5",
        ]
        predictions = []
        reasonings = []

        for model in models:
            original_default = self._llms.get("default")
            original_parser = self._llms.get("parser")
            self._llms["default"] = GeneralLlm(model=model)
            self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-4.1-mini")

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
            finally:
                self._llms["default"] = original_default
                self._llms["parser"] = original_parser

        # Median aggregation
        if isinstance(question, BinaryQuestion):
            median_val = median([p for p in predictions])
            return ReasonedPrediction(prediction_value=median_val, reasoning=" | ".join(reasonings))

        elif isinstance(question, MultipleChoiceQuestion):
            options = question.options
            avg_probs = {}
            for opt in options:
                probs = [ {po.option_name: po.probability for po in p.predicted_options}.get(opt, 0.0) for p in predictions ]
                avg_probs[opt] = median(probs)
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
                vals = []
                for dist in predictions:
                    for item in dist.declared_percentiles:
                        if abs(item.percentile - pt) < 0.01:
                            vals.append(item.value)
                            break
                median_pcts.append(Percentile(percentile=pt, value=median(vals) if vals else 0.0))
            final_dist = NumericDistribution.from_question(median_pcts, question)
            return ReasonedPrediction(prediction_value=final_dist, reasoning=" | ".join(reasonings))

        else:
            return ReasonedPrediction(prediction_value=predictions[0], reasoning=" | ".join(reasonings))

    # -----------------------------
    # POST-RUN
    # -----------------------------
    async def _compute_brier_scores(self):
        try:
            api = MetaculusApi()
            binary_records = [r for r in self._prediction_records if r["type"] == "BinaryQuestion" and r["predicted_prob"] is not None]
            question_ids = [r["question_id"] for r in binary_records if isinstance(r["question_id"], (int, str)) and r["question_id"] not in ("N/A", "unknown")]
            if not question_ids:
                return
            resolved_qs = await api.get_questions_by_ids(question_ids)
            brier_sum, scored = 0.0, 0
            for q in resolved_qs:
                if isinstance(q, BinaryQuestion) and q.resolution:
                    rec = next((r for r in binary_records if r["question_id"] == q.id), None)
                    if rec:
                        pred = rec["predicted_prob"]
                        actual = 1.0 if q.resolution == "yes" else 0.0
                        brier = (pred - actual) ** 2
                        brier_sum += brier
                        scored += 1
                        rec.update({"resolution": q.resolution, "actual": actual, "brier_score": round(brier, 4)})
            if scored:
                logger.info(f"📊 Avg Brier (n={scored}): {brier_sum / scored:.4f}")
        except Exception as e:
            logger.error(f"Brier failed: {e}")

    def export_predictions_to_csv(self, filepath: str = "upskill_bot_forecasts.csv"):
        if not self._prediction_records:
            return
        df = pd.DataFrame([{k: v for k, v in r.items() if isinstance(v, (str, int, float, bool, type(None)))} for r in self._prediction_records])
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

    # Optional auth
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
            "default": GeneralLlm(model="openrouter/openai/gpt-5.1", temperature=0.2),
            "parser": GeneralLlm(model="openrouter/openai/gpt-4.1-mini", temperature=0.0),
            "researcher_gpt": GeneralLlm(model="openrouter/openai/gpt-5", temperature=0.3),
            "researcher_claude": GeneralLlm(model="openrouter/anthropic/claude-sonnet-4.5", temperature=0.3),
            "summarizer": GeneralLlm(model="openrouter/openai/gpt-4.1-mini", temperature=0.0),  # ✅ Explicitly set
        },
    )

    tournament_ids = [32916, "ACX2026", "minibench"]
    logger.info("🚀 Starting UpskillBot (GPT-5/5.1 + Claude Sonnet 4.5)...")
    asyncio.run(bot.run_all_tournaments(tournament_ids))
    logger.info("🏁 UpskillBot run completed successfully.")
