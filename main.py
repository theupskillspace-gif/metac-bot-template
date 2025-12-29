from __future__ import annotations  # ✅ Critical: delays annotation evaluation

import argparse
import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import pandas as pd
from tavily import TavilyClient

# 🔴 SAFETY: Import early and validate required types
try:
    from forecasting_tools import (
        BinaryQuestion,
        ForecastBot,
        GeneralLlm,
        MetaculusApi,
        MetaculusQuestion,
        MultipleChoiceQuestion,
        NumericDistribution,
        NumericQuestion,      # ✅ Must be present
        Percentile,
        BinaryPrediction,
        PredictedOptionList,
        ReasonedPrediction,
        clean_indents,
        structure_output,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import from 'forecasting_tools'. "
        "Ensure the package is installed and contains: "
        "BinaryQuestion, NumericQuestion, MultipleChoiceQuestion, etc. "
        f"Error: {e}"
    )

# Double-check required types are available (debug safeguard)
_required_types = ["NumericQuestion", "BinaryQuestion", "MultipleChoiceQuestion"]
for t in _required_types:
    if t not in globals():
        raise NameError(f"Critical type '{t}' not imported from forecasting_tools!")

logger = logging.getLogger("UpskillBot")


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

    async def _tavily_search_limited(self, query: str) -> dict:
        async with self._tavily_lock:
            if self._tavily_query_count >= self._max_tavily_queries:
                raise RuntimeError(
                    f"UpskillBot: Tavily query limit reached ({self._max_tavily_queries})."
                )
            self._tavily_query_count += 1
            count = self._tavily_query_count
            logger.debug(f"[Tavily Query #{count}] {query[:100]}...")

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.tavily.search(
                query=query.strip(),
                search_depth="advanced",
                include_answer=True,
                max_results=5,
                days=180,
            ),
        )
        return response

    def _is_stock_question(self, question: MetaculusQuestion) -> bool:
        text = " ".join([
            question.question_text,
            question.background_info or "",
            question.resolution_criteria or "",
        ]).lower()
        patterns = [
            r"\b(?:stock|equity|share|s&p|nasdaq|dow|djia|index|ticker)\b",
            r"\b[a-z]{1,4}\.?\s*\d{1,2}/\d{1,2}",
            r"\b\$?[a-z]{1,5}\b",
        ]
        return any(re.search(pat, text) for pat in patterns)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            try:
                query = (
                    f"Forecasting {question.question_text}. "
                    f"Resolution: {question.resolution_criteria}. "
                    f"Context: {question.background_info or ''}"
                )[:300]

                response = await self._tavily_search_limited(query)

                answer = response.get("answer", "").strip() or "[No summary]"
                results = response.get("results", [])
                snippets = "\n\n".join(
                    [
                        f"[{i+1}] {res['title'][:80]} — {res['content'][:400]}..."
                        for i, res in enumerate(results[:4])
                    ]
                )

                return clean_indents(
                    f"""
                    ### UpskillBot Real-Time Research (as of {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})
                    **AI Summary**: {answer}
                    **Top Sources**:
                    {snippets if snippets else 'None retrieved.'}
                    🔍 Query used: "{query}"
                    """
                )
            except Exception as e:
                logger.error(f"UpskillBot: Research failed for Q{question.id}: {e}")
                return f"⚠️ Research unavailable ({e}). Use caution."

    def _apply_double_median_to_numeric(self, dist: NumericDistribution) -> NumericDistribution:
        pcts = {p.percentile: p.value for p in dist.declared_percentiles}
        needed = [10, 20, 40, 50, 60, 80, 90]

        if 50 not in pcts:
            if 40 in pcts and 60 in pcts:
                pcts[50] = (pcts[40] + pcts[60]) / 2
            elif 20 in pcts and 80 in pcts:
                pcts[50] = (pcts[20] + pcts[80]) / 2
            else:
                pcts[50] = (min(pcts.values()) + max(pcts.values())) / 2

        median = pcts[50]
        new_pcts = {}
        for p in needed:
            if p in pcts:
                shrink = 0.3 if abs(p - 50) >= 30 else 0.1
                new_pcts[p] = (1 - shrink) * pcts[p] + shrink * median
            else:
                lower = max([k for k in pcts.keys() if k <= p], default=min(pcts.keys()))
                upper = min([k for k in pcts.keys() if k >= p], default=max(pcts.keys()))
                if lower == upper:
                    new_pcts[p] = pcts[lower]
                else:
                    w = (p - lower) / (upper - lower)
                    new_pcts[p] = (1 - w) * pcts[lower] + w * pcts[upper]

        final_percentiles = [
            Percentile(percentile=p, value=new_pcts[p])
            for p in [10, 20, 40, 60, 80, 90]
        ]
        return NumericDistribution.from_question(final_percentiles, dist.question)

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = self._build_binary_prompt(question, research)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        try:
            pred: BinaryPrediction = await structure_output(
                reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
            )
            prob = max(0.01, min(0.99, pred.prediction_in_decimal))
        except Exception as e:
            logger.warning(f"UpskillBot: Binary parse fail for Q{question.id}: {e}")
            prob = 0.5

        self._record_prediction(question, prob, reasoning)
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    def _build_binary_prompt(self, question: BinaryQuestion, research: str) -> str:
        is_stock = self._is_stock_question(question)
        stock_note = (
            "\n⚠️ Stock/market question: short-term moves are mostly noise — avoid overconfidence."
            if is_stock
            else ""
        )
        return clean_indents(
            f"""
            UpskillBot superforecaster. Use only real evidence.

            Question: {question.question_text}
            Resolution: {question.resolution_criteria}
            Research:
            {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}{stock_note}

            → Be humble. Unless evidence is overwhelming, keep 5% ≤ P ≤ 95%.
            Final line: "Probability: ZZ%"
            """
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = self._build_mc_prompt(question, research)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        try:
            pred_list = await structure_output(
                reasoning,
                PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=f"Options: {question.options}",
            )
        except Exception as e:
            logger.warning(f"UpskillBot: MC parse fail for Q{question.id}: {e}")
            n = len(question.options)
            pred_list = PredictedOptionList(
                probabilities={opt: round(100.0 / n, 1) for opt in question.options}
            )

        top_opt = max(pred_list.probabilities, key=pred_list.probabilities.get)
        top_prob = pred_list.probabilities[top_opt] / 100.0
        self._record_prediction(question, top_prob, reasoning, extra={"top_option": top_opt})
        return ReasonedPrediction(prediction_value=pred_list, reasoning=reasoning)

    def _build_mc_prompt(self, question: MultipleChoiceQuestion, research: str) -> str:
        return clean_indents(
            f"""
            UpskillBot mode. Options: {question.options}
            Research:
            {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}
            Assign probabilities summing to 100%. End with:
            Option_A: 30.0
            Option_B: 70.0
            """
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str  # ✅ Safe now with `from __future__ import annotations`
    ) -> ReasonedPrediction[NumericDistribution]:
        prompt = self._build_numeric_prompt(question, research)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        try:
            pct_list: List[Percentile] = await structure_output(
                reasoning, list[Percentile], model=self.get_llm("parser", "llm")
            )
            dist = NumericDistribution.from_question(pct_list, question)

            if self._is_stock_question(question):
                logger.info(f"UpskillBot: Double-median on stock Q{question.id}")
                dist = self._apply_double_median_to_numeric(dist)

        except Exception as e:
            logger.warning(f"UpskillBot: Numeric parse fail for Q{question.id}: {e}")
            lo, hi = question.lower_bound, question.upper_bound
            fallback = [Percentile(p, lo + (hi - lo) * p / 100) for p in [10, 20, 40, 60, 80, 90]]
            dist = NumericDistribution.from_question(fallback, question)

        median = dist.get_percentile_value(50)
        self._record_prediction(question, None, reasoning, extra={"median": median})
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    def _build_numeric_prompt(self, question: NumericQuestion, research: str) -> str:
        lo = question.nominal_lower_bound or question.lower_bound
        hi = question.nominal_upper_bound or question.upper_bound
        lo_msg = f"Lower bound: {'≥' if question.open_lower_bound else ''}{lo}"
        hi_msg = f"Upper bound: {'≤' if question.open_upper_bound else ''}{hi}"
        is_stock = self._is_stock_question(question)
        stock_note = "\n⚠️ Stock question: use conservative, mean-reverting ranges." if is_stock else ""

        return clean_indents(
            f"""
            UpskillBot numeric forecaster.

            Question: {question.question_text}
            {lo_msg}
            {hi_msg}{stock_note}

            Research:
            {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            Output EXACTLY:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
            """
        )

    def _record_prediction(
        self,
        question: MetaculusQuestion,
        prob: Optional[float],
        reasoning: str,
        extra: Optional[Dict] = None,
    ):
        record = {
            "question_id": question.id,
            "page_url": question.page_url,
            "title": question.question_text[:100],
            "type": question.get_question_type().__name__,
            "predicted_prob": prob,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "tavily_queries_used": self._tavily_query_count,
            "is_stock": self._is_stock_question(question),
            "reasoning_snippet": reasoning[:500].replace("\n", " "),
        }
        if extra:
            record.update(extra)
        self._prediction_records.append(record)

    async def _compute_brier_scores(self):
        try:
            api = MetaculusApi()
            binary_records = [
                r for r in self._prediction_records
                if r["type"] == "BinaryQuestion" and r["predicted_prob"] is not None
            ]
            if not binary_records:
                return

            question_ids = [r["question_id"] for r in binary_records]
            resolved_questions = await api.get_questions_by_ids(question_ids)

            brier_sum = 0.0
            scored = 0
            for q in resolved_questions:
                if not isinstance(q, BinaryQuestion) or q.resolution is None:
                    continue
                pred_rec = next((r for r in binary_records if r["question_id"] == q.id), None)
                if not pred_rec:
                    continue

                pred = pred_rec["predicted_prob"]
                actual = 1.0 if q.resolution == "yes" else 0.0
                brier = (pred - actual) ** 2
                brier_sum += brier
                scored += 1
                pred_rec.update({"resolution": q.resolution, "actual": actual, "brier_score": round(brier, 4)})

            if scored > 0:
                avg_brier = brier_sum / scored
                logger.info(f"📊 UpskillBot Avg Brier (n={scored}): {avg_brier:.4f}")

        except Exception as e:
            logger.error(f"UpskillBot: Brier computation failed: {e}")

    def export_predictions_to_csv(self, filepath: str = "upskill_bot_forecasts.csv"):
        if not self._prediction_records:
            logger.warning("UpskillBot: No predictions to export.")
            return

        df = pd.DataFrame([
            {k: v for k, v in r.items() if isinstance(v, (str, int, float, bool, type(None)))}
            for r in self._prediction_records
        ])
        df.to_csv(filepath, index=False)
        logger.info(f"✅ UpskillBot exported {len(df)} predictions to {filepath}")

    async def run_all_tournaments(self, tournament_ids: List):
        all_reports = []
        for tid in tournament_ids:
            try:
                logger.info(f"▶ UpskillBot: Forecasting tournament '{tid}'")
                reports = await self.forecast_on_tournament(tid, return_exceptions=True)
                all_reports.extend(reports)
            except Exception as e:
                logger.error(f"💥 UpskillBot: Tournament '{tid}' failed: {e}")

        await self._compute_brier_scores()
        self.export_predictions_to_csv()
        self.log_report_summary(all_reports)


# ===== MAIN =====
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # Optional auth
    email = os.getenv("METACULUS_EMAIL")
    password = os.getenv("METACULUS_PASSWORD")
    token = os.getenv("METACULUS_TOKEN")

    if token or (email and password):
        try:
            api = MetaculusApi()
            api.login_with_token(token) if token else api.login(email, password)
            logger.info("✅ UpskillBot: Metaculus auth successful.")
        except Exception as e:
            logger.error(f"UpskillBot: Metaculus login failed: {e}")

    # 🤖 Launch
    GPT5 = "openrouter/openai/gpt-5"
    bot = UpskillBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        publish_reports_to_metaculus=bool(token or (email and password)),
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model=GPT5, temperature=0.2, timeout=90),
            "parser": GeneralLlm(model=GPT5, temperature=0.0, timeout=60),
        },
    )

    tournament_ids = [32916, "ACX2026", "minibench"]
    logger.info("🚀 Starting UpskillBot forecast run...")
    asyncio.run(bot.run_all_tournaments(tournament_ids))
    logger.info("🏁 UpskillBot run completed successfully.")
