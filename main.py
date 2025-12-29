import argparse
import asyncio
import csv
import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import pandas as pd
from tavily import TavilyClient

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)


logger = logging.getLogger(__name__)


class TavilyResearchBot2025(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY must be set.")
        self.tavily = TavilyClient(api_key=api_key)

        # ⏱️ Global Tavily query counter + limit
        self._tavily_query_count = 0
        self._max_tavily_queries = 400
        self._tavily_lock = asyncio.Lock()

        # 📊 For Brier & CSV export
        self._prediction_records: List[Dict[str, Any]] = []

    # ===== TAVILY WITH RATE LIMITING =====
    async def _tavily_search_limited(self, query: str) -> dict:
        async with self._tavily_lock:
            if self._tavily_query_count >= self._max_tavily_queries:
                raise RuntimeError(
                    f"Tavily query limit reached ({self._max_tavily_queries}). "
                    "Stopping further research to avoid overage."
                )
            self._tavily_query_count += 1
            count = self._tavily_query_count
            logger.debug(f"[Tavily Query #{count}] {query[:100]}...")

        # Run sync Tavily in thread pool to avoid blocking
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

    # ===== STOCK DETECTION =====
    def _is_stock_question(self, question: MetaculusQuestion) -> bool:
        text = " ".join([
            question.question_text,
            question.background_info or "",
            question.resolution_criteria or "",
        ]).lower()
        stock_patterns = [
            r"\b(?:stock|equity|share|s&p|nasdaq|dow|djia|index|ticker|nyse|nasdaq)\b",
            r"\b[a-z]{1,4}\.?\s*\d{1,2}/\d{1,2}",
            r"\b(?:\$?[a-z]{1,5}\b)",  # crude ticker check (e.g., AAPL, TSLA)
        ]
        return any(re.search(pat, text) for pat in stock_patterns)

    # ===== RESEARCH =====
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            try:
                # Build focused query (truncate to avoid token/length issues)
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
                    ### Tavily Research (as of {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})
                    **AI Summary**: {answer}

                    **Top Sources**:
                    {snippets if snippets else 'None retrieved.'}

                    🔍 Query used: "{query}"
                    """
                )

            except Exception as e:
                logger.error(f"Tavily failed for {question.id}: {e}")
                return f"⚠️ Research unavailable ({e}). Use caution."

    # ===== FORECAST HELPERS =====
    def _apply_double_median_to_numeric(self, dist: NumericDistribution) -> NumericDistribution:
        """
        For stock/index questions: apply double median stabilization.
        Reduces sensitivity to extreme bullish/bearish outliers.
        """
        # Get percentiles as dict
        pcts = {p.percentile: p.value for p in dist.declared_percentiles}
        needed = [10, 20, 40, 50, 60, 80, 90]

        # Ensure 50th (median) exists
        if 50 not in pcts:
            # Interpolate median
            if 40 in pcts and 60 in pcts:
                pcts[50] = (pcts[40] + pcts[60]) / 2
            elif 20 in pcts and 80 in pcts:
                pcts[50] = (pcts[20] + pcts[80]) / 2
            else:
                pcts[50] = (min(pcts.values()) + max(pcts.values())) / 2

        # First median: center around 50th
        median = pcts[50]
        # Second median: shrink tails toward median (robustify)
        new_pcts = {}
        for p in needed:
            if p in pcts:
                # Pull extreme percentiles toward median
                shrink = 0.3 if abs(p - 50) >= 30 else 0.1  # 10/90 shrink more
                new_val = (1 - shrink) * pcts[p] + shrink * median
                new_pcts[p] = new_val
            else:
                # Interpolate missing
                lower = max([k for k in pcts.keys() if k <= p], default=min(pcts.keys()))
                upper = min([k for k in pcts.keys() if k >= p], default=max(pcts.keys()))
                if lower == upper:
                    new_pcts[p] = pcts[lower]
                else:
                    w = (p - lower) / (upper - lower)
                    new_pcts[p] = (1 - w) * pcts[lower] + w * pcts[upper]

        # Rebuild percentile list (keep only standard ones)
        final_percentiles = [
            Percentile(percentile=p, value=new_pcts[p])
            for p in [10, 20, 40, 60, 80, 90]
        ]
        return NumericDistribution.from_question(final_percentiles, dist.question)

    # ===== CORE FORECAST METHODS =====
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
            logger.warning(f"Binary parse fail {question.id}: {e}")
            prob = 0.5

        # Store for Brier & CSV
        self._record_prediction(question, prob, reasoning)

        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    def _build_binary_prompt(self, question: BinaryQuestion, research: str) -> str:
        is_stock = self._is_stock_question(question)
        stock_note = (
            "\n⚠️ This appears to be a stock/market question. Base rates: most short-term market moves are noise; avoid overconfidence."
            if is_stock
            else ""
        )
        return clean_indents(
            f"""
            You are a calibration-aware superforecaster. Use only real evidence.

            Question: {question.question_text}
            Resolution: {question.resolution_criteria}
            Background: {question.background_info}
            {question.fine_print}

            Research:
            {research}

            Today: {datetime.now().strftime('%Y-%m-%d')}{stock_note}

            Analysis:
            (a) Time to resolution? (Short-term → more noise; long-term → more signal)
            (b) Base rate (e.g., % of similar bills passed, startups succeeded, etc.)
            (c) Status quo trajectory (most things don’t change abruptly)
            (d) What concrete evidence would push probability >80% or <20%?
            (e) Am I over/under-confident? (95% of forecasters are overconfident.)

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
            logger.warning(f"MC parse fail {question.id}: {e}")
            n = len(question.options)
            pred_list = PredictedOptionList(
                probabilities={opt: round(100.0 / n, 1) for opt in question.options}
            )

        # Store as entropy or top prob — for CSV, log top
        top_opt = max(pred_list.probabilities, key=pred_list.probabilities.get)
        top_prob = pred_list.probabilities[top_opt] / 100.0
        self._record_prediction(question, top_prob, reasoning, extra={"top_option": top_opt})

        return ReasonedPrediction(prediction_value=pred_list, reasoning=reasoning)

    def _build_mc_prompt(self, question: MultipleChoiceQuestion, research: str) -> str:
        return clean_indents(
            f"""
            Superforecaster mode. Ground in research.

            Question: {question.question_text}
            Options: {question.options}
            Resolution: {question.resolution_criteria}
            Background: {question.background_info}

            Research:
            {research}

            Today: {datetime.now().strftime('%Y-%m-%d')}

            Guidelines:
            - Assign ≥1% to plausible surprises.
            - Sum to 100%.
            - If uncertain, flatten probabilities.

            End with exact format:
            Option_A: 30.0
            Option_B: 70.0
            """
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        prompt = self._build_numeric_prompt(question, research)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        try:
            pct_list: List[Percentile] = await structure_output(
                reasoning, list[Percentile], model=self.get_llm("parser", "llm")
            )
            dist = NumericDistribution.from_question(pct_list, question)

            # 🔁 Double-median for stock questions
            if self._is_stock_question(question):
                logger.info(f"Applying double-median adjustment (stock question) to {question.id}")
                dist = self._apply_double_median_to_numeric(dist)

        except Exception as e:
            logger.warning(f"Numeric parse fail {question.id}: {e}")
            lo, hi = question.lower_bound, question.upper_bound
            fallback = [
                Percentile(p, lo + (hi - lo) * p / 100)
                for p in [10, 20, 40, 60, 80, 90]
            ]
            dist = NumericDistribution.from_question(fallback, question)

        # Log median for CSV
        median = dist.get_percentile_value(50)
        self._record_prediction(question, None, reasoning, extra={"median": median})

        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    def _build_numeric_prompt(self, question: NumericQuestion, research: str) -> str:
        lo_msg, hi_msg = self._create_bound_messages(question)
        is_stock = self._is_stock_question(question)
        stock_note = (
            "\n⚠️ This is likely a stock/market question. Use conservative ranges; markets are volatile but mean-reverting short-term."
            if is_stock
            else ""
        )
        return clean_indents(
            f"""
            Calibrated numeric forecaster. Cite numbers from research.

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Infer'}
            Resolution: {question.resolution_criteria}
            {lo_msg}
            {hi_msg}{stock_note}

            Research:
            {research}

            Today: {datetime.now().strftime('%Y-%m-%d')}

            Steps:
            (a) Base rate / historical value
            (b) Current trend (e.g., +5%/yr?)
            (c) Expert consensus (if any)
            (d) Low scenario (10th %ile): what breaks?
            (e) High scenario (90th %ile): what accelerates?
            (f) Are tails fat? (e.g., black swans → widen CI)

            → Be humble: 90% CI should often span 3–10x for open-ended estimates.

            Output EXACTLY:
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            """
        )

    def _create_bound_messages(self, q) -> tuple[str, str]:
        lo = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        hi = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        lo_msg = f"Lower bound: {'≥' if q.open_lower_bound else ''}{lo}"
        hi_msg = f"Upper bound: {'≤' if q.open_upper_bound else ''}{hi}"
        return lo_msg, hi_msg

    # ===== BRIER & RECORDING =====
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
        """
        Fetch resolved binary questions and compute Brier scores.
        Requires Metaculus auth.
        """
        try:
            api = MetaculusApi()
            # Get all binary questions we forecasted that are now resolved
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
                # Find prediction
                pred_rec = next((r for r in binary_records if r["question_id"] == q.id), None)
                if not pred_rec:
                    continue

                pred = pred_rec["predicted_prob"]
                actual = 1.0 if q.resolution == "yes" else 0.0
                brier = (pred - actual) ** 2
                brier_sum += brier
                scored += 1

                # Update record
                pred_rec["resolution"] = q.resolution
                pred_rec["actual"] = actual
                pred_rec["brier_score"] = round(brier, 4)

            if scored > 0:
                avg_brier = brier_sum / scored
                logger.info(f"📊 Avg Brier score (n={scored}): {avg_brier:.4f}")
                for rec in self._prediction_records:
                    if "brier_score" in rec:
                        logger.info(
                            f"  Q{rec['question_id']}: P={rec['predicted_prob']:.2%} → {rec['resolution']} | Brier={rec['brier_score']}"
                        )

        except Exception as e:
            logger.error(f"Brier computation failed: {e}")

    # ===== POST-RUN OUTPUT =====
    def export_predictions_to_csv(self, filepath: str = "forecasts_output.csv"):
        if not self._prediction_records:
            logger.warning("No predictions to export.")
            return

        # Normalize records into flat dict
        flat_records = []
        for r in self._prediction_records:
            flat = {k: v for k, v in r.items() if isinstance(v, (str, int, float, bool, type(None)))}
            flat_records.append(flat)

        df = pd.DataFrame(flat_records)
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Exported {len(df)} predictions to {filepath}")

        # Optional: log summary stats
        binary = df[df["type"] == "BinaryQuestion"]
        if not binary.empty:
            avg_prob = binary["predicted_prob"].mean()
            brier_avg = binary["brier_score"].mean() if "brier_score" in binary else "N/A"
            logger.info(f"  Binary: {len(binary)} questions, avg predicted prob = {avg_prob:.2%}, avg Brier = {brier_avg}")

    async def forecast_on_tournament(self, tournament_id, **kwargs):
        # Wrap to ensure post-processing
        reports = await super().forecast_on_tournament(tournament_id, **kwargs)
        return reports

    async def run_all_tournaments(self, tournament_ids: List):
        all_reports = []
        for tid in tournament_ids:
            try:
                logger.info(f"▶ Forecasting tournament: {tid}")
                reports = await self.forecast_on_tournament(tid, return_exceptions=True)
                all_reports.extend(reports)
                logger.info(f"✅ Done with {tid}")
            except Exception as e:
                logger.error(f"💥 Tournament {tid} failed: {e}")

        # Post-run: Brier + CSV
        await self._compute_brier_scores()
        self.export_predictions_to_csv()
        self.log_report_summary(all_reports)


# ===== MAIN =====
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # 🔑 Metaculus auth (for publishing & Brier)
    email = os.getenv("METACULUS_EMAIL")
    password = os.getenv("METACULUS_PASSWORD")
    token = os.getenv("METACULUS_TOKEN")

    if not (token or (email and password)):
        logger.warning("No Metaculus credentials — publishing & Brier scores disabled.")
    else:
        try:
            api = MetaculusApi()
            if token:
                api.login_with_token(token)
            else:
                api.login(email, password)
            logger.info("✅ Metaculus auth successful.")
        except Exception as e:
            logger.error(f"Metaculus login failed: {e}")

    # 🤖 Bot setup
    GPT5 = "openrouter/openai/gpt-5"
    bot = TavilyResearchBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,  # will skip if unauthenticated
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(model=GPT5, temperature=0.2, timeout=90),
            "parser": GeneralLlm(model=GPT5, temperature=0.0, timeout=60),
        },
    )

    # ✅ Your exact tournament IDs
    tournament_ids = [32916, "ACX2026", "minibench"]

    # 🚀 Run
    asyncio.run(bot.run_all_tournaments(tournament_ids))
