from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import textwrap
import time
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
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

# =========================================================
# ⚙️ Configuration
# =========================================================
DEFAULT_FORECASTER = "openrouter/anthropic/claude-sonnet-4.5"
PARSER_MODEL = "openrouter/openai/gpt-4.1-mini"
SUMMARIZER_MODEL = "openrouter/openai/gpt-4.1-mini"

RESEARCH_TIMEOUT_S = float(os.getenv("RESEARCH_TIMEOUT_S", "25"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "70"))
MAX_CONCURRENT_QUESTIONS = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "1"))
RETRY_MAX = int(os.getenv("RETRY_MAX", "6"))
RETRY_BASE_S = float(os.getenv("RETRY_BASE_S", "2.0"))
RETRY_MAX_S = float(os.getenv("RETRY_MAX_S", "60.0"))

MIN_P = float(os.getenv("MIN_P", "0.01"))
MAX_P = float(os.getenv("MAX_P", "0.99"))

CALIBRATION_LOG_FILE = "upskill_calibration_log.jsonl"
PREDICTIONS_CSV_FILE = "upskill_bot_forecasts.csv"
COSTS_CSV_FILE = "upskill_bot_costs.csv"

# =========================================================
# 🔍 Exa Searcher
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
# ✅ Normalized Multiple Choice Model (Pydantic v2-safe)
# =========================================================
class NormalizedPredictedOptionList(PredictedOptionList):
    @model_validator(mode="after")
    def normalize_probs(self):
        if not getattr(self, "predicted_options", None):
            return self
        
        probs = [float(p.probability) for p in self.predicted_options]
        total = sum(probs)

        if total <= 0:
            logger.warning(f"PredictedOptionList sum is {total}. Raw: {self.predicted_options}")
            return self

        # If model returned percentages (e.g., 45, 30, 25)
        if any(p > 1.0 for p in probs) or total > 1.5:
            for opt in self.predicted_options:
                opt.probability = float(opt.probability) / 100.0

        # Normalize to sum to 1
        total = sum(float(p.probability) for p in self.predicted_options)
        if total > 0 and abs(total - 1.0) > 1e-6:
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


# =========================================================
# 🧠 Archetypes (Forecasting Structure Types)
# =========================================================
class Archetype(str, Enum):
    MARKET = "market"
    ELECTION = "election_politics"
    SCI_TECH = "sci_tech"
    MACRO = "macro"
    GEO_CONFLICT = "geo_conflict"
    ORG_PRODUCT = "org_product"
    WEATHER_NATURAL = "weather_natural"
    SPORTS_ENT = "sports_entertainment"
    OTHER = "other"


@dataclass(frozen=True)
class ArchetypeTuning:
    # Extremization aggressiveness (max k scaling); reliability bands decide exact k
    max_k_binary: float
    max_k_mc: float
    # Binary clipping bounds
    clip_lo: float
    clip_hi: float
    # Research hints
    recent_suffix: str
    historical_suffix: str


def archetype_tuning(a: Archetype) -> ArchetypeTuning:
    # Conservative-but-bold defaults; market/sports more extremizable, geo/other less.
    if a == Archetype.MARKET:
        return ArchetypeTuning(
            max_k_binary=2.6,
            max_k_mc=2.5,
            clip_lo=0.005,
            clip_hi=0.995,
            recent_suffix="latest catalysts, earnings/guidance, analyst consensus, comparable historical moves",
            historical_suffix="historical base rates, comparable events, long-run frequencies for similar market outcomes",
        )
    if a == Archetype.SPORTS_ENT:
        return ArchetypeTuning(
            max_k_binary=2.4,
            max_k_mc=2.3,
            clip_lo=0.01,
            clip_hi=0.99,
            recent_suffix="odds, injuries/rosters, schedule, credible reporting, recent form",
            historical_suffix="historical base rates for similar matchups/awards/releases; comparable seasons",
        )
    if a == Archetype.WEATHER_NATURAL:
        return ArchetypeTuning(
            max_k_binary=2.0,
            max_k_mc=1.9,
            clip_lo=0.01,
            clip_hi=0.99,
            recent_suffix="official agencies, warnings/advisories, near-term forecasts where applicable",
            historical_suffix="historical frequencies, seasonality, reference class base rates for similar events",
        )
    if a == Archetype.ELECTION:
        return ArchetypeTuning(
            max_k_binary=2.2,
            max_k_mc=2.1,
            clip_lo=0.01,
            clip_hi=0.99,
            recent_suffix="polling averages, fundamentals, election models, recent events affecting turnout/coalitions",
            historical_suffix="incumbency/base-rate priors, polling error distributions, comparable elections",
        )
    if a == Archetype.MACRO:
        return ArchetypeTuning(
            max_k_binary=2.1,
            max_k_mc=2.0,
            clip_lo=0.01,
            clip_hi=0.99,
            recent_suffix="latest official releases, consensus forecasts, central bank signals, leading indicators",
            historical_suffix="historical distributions, seasonality, forecast error norms for similar series",
        )
    if a == Archetype.SCI_TECH:
        return ArchetypeTuning(
            max_k_binary=2.1,
            max_k_mc=2.0,
            clip_lo=0.01,
            clip_hi=0.99,
            recent_suffix="papers, benchmarks, roadmaps, releases, credible expert commentary, timelines",
            historical_suffix="rate-of-progress reference classes; similar milestone timelines; base rates",
        )
    if a == Archetype.ORG_PRODUCT:
        return ArchetypeTuning(
            max_k_binary=2.1,
            max_k_mc=2.0,
            clip_lo=0.01,
            clip_hi=0.99,
            recent_suffix="official statements, filings, roadmap signals, hiring/funding, credible reporting",
            historical_suffix="historical launch cadence, similar org decisions, base rates of comparable events",
        )
    if a == Archetype.GEO_CONFLICT:
        return ArchetypeTuning(
            max_k_binary=1.9,
            max_k_mc=1.8,
            clip_lo=0.02,
            clip_hi=0.98,
            recent_suffix="credible reporting, official statements, timelines, actions vs rhetoric, comparable crises",
            historical_suffix="reference class of similar conflicts/diplomatic events; base rates; escalation frequencies",
        )
    return ArchetypeTuning(
        max_k_binary=2.0,
        max_k_mc=1.9,
        clip_lo=0.01,
        clip_hi=0.99,
        recent_suffix="recent developments, credible reporting, and domain-relevant indicators",
        historical_suffix="historical base rates / reference class / long-run frequencies for similar events",
    )


# =========================================================
# 🧰 Helpers
# =========================================================
def build_tavily_query(question: MetaculusQuestion, max_chars: int = 300) -> str:
    q = (question.question_text or " ").strip()
    bg = (question.background_info or " ").strip()
    q = re.sub(r"http\S+", " ", q)
    bg = re.sub(r"http\S+", " ", bg)
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
    return float(max(MIN_P, min(MAX_P, p)))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def logit(p: float, eps: float = 1e-9) -> float:
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def conservative_clip(p: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, p))


def extract_probability(text: str) -> Optional[float]:
    patterns = [
        r"Probability\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"\bProb(?:ability)?\b\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"\bP\b\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*%\s*$",
        r"\b([01](?:\.[0-9]+)?)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        val = float(m.group(1))
        if val > 1.0:
            return clamp01(val / 100.0)
        return clamp01(val)
    return None


def shorten_reasoning(text: str, max_chars: int = 900) -> str:
    t = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def stable_question_cache_key(question: MetaculusQuestion) -> str:
    qid = getattr(question, "id", None) or getattr(question, "question_id", None)
    if qid is not None:
        return str(qid)
    payload = (question.question_text or " ") + "||" + (question.background_info or " ") + "||" + (
        question.resolution_criteria or " "
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def canonicalize_option_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip())


def map_to_valid_options(pred: NormalizedPredictedOptionList, valid_options: List[str]) -> NormalizedPredictedOptionList:
    valid_norm = {canonicalize_option_name(o): o for o in valid_options}
    seen: Dict[str, float] = {}
    for opt in pred.predicted_options:
        norm = canonicalize_option_name(opt.option_name)
        if norm in valid_norm:
            seen[valid_norm[norm]] = float(opt.probability)
    
    # Ensure all options present; missing get 0.0 then normalize
    full_probs = [max(0.0, float(seen.get(o, 0.0))) for o in valid_options]
    s = sum(full_probs)
    if s <= 0:
        p = 1.0 / max(1, len(valid_options))
        full_probs = [p for _ in valid_options]
    else:
        full_probs = [p / s for p in full_probs]

    return NormalizedPredictedOptionList(
        predicted_options=[PredictedOption(option_name=o, probability=float(full_probs[i])) for i, o in enumerate(valid_options)]
    )


def _normalize_predicted_option(raw: dict) -> dict:
    """Ensure option dict uses 'option_name' field for Pydantic compatibility."""
    result = raw.copy()
    if "option" in result and "option_name" not in result:
        result["option_name"] = result.pop("option")
    return result


# =========================================================
# 🧠 Archetype Detection (Rules First, LLM Fallback)
# =========================================================
def _question_text_blob(q: MetaculusQuestion) -> str:
    return " ".join([q.question_text or "", q.background_info or "", q.resolution_criteria or ""]).lower()


def rule_archetype(text: str) -> tuple[Optional[Archetype], float]:
    # MARKET
    if re.search(r"\b(stock|shares?|equity|ticker|nasdaq|nyse|s&p|dow|crypto|bitcoin|eth|price target|bond yield)\b", text):
        return Archetype.MARKET, 0.95
    # ELECTION/POLITICS
    if re.search(r"\b(election|polls?|parliament|senate|congress|president|prime minister|referendum|bill|law|legislation)\b", text):
        return Archetype.ELECTION, 0.85
    # MACRO
    if re.search(r"\b(cpi|inflation|gdp|unemployment|interest rate|fed|ecb|boj|boe|central bank|ppi)\b", text):
        return Archetype.MACRO, 0.85
    # GEO/CONFLICT
    if re.search(r"\b(war|invasion|ceasefire|missile|sanctions?|nato|military|conflict|hostages?|border clash)\b", text):
        return Archetype.GEO_CONFLICT, 0.80
    # WEATHER/NATURAL
    if re.search(r"\b(hurricane|typhoon|earthquake|volcano|wildfire|storm|flood|tornado)\b", text):
        return Archetype.WEATHER_NATURAL, 0.85
    # SPORTS/ENT
    if re.search(r"\b(championship|tournament|world cup|nba|nfl|mlb|nhl|ucl|olympics|oscars?|grammys?|bafta|emmys?)\b", text):
        return Archetype.SPORTS_ENT, 0.80
    # SCI/TECH
    if re.search(r"\b(benchmark|model|ai|paper|arxiv|launch|prototype|clinical trial|phase i|phase ii|phase iii|compute|chip)\b", text):
        return Archetype.SCI_TECH, 0.75
    # ORG/PRODUCT
    if re.search(r"\b(acquisition|merger|funding|series [a-z]|ipo|product|policy update|terms of service|reorg|layoffs?)\b", text):
        return Archetype.ORG_PRODUCT, 0.70
    return None, 0.0


# =========================================================
# 🎯 Reliability + Extremization (Type-Aware)
# =========================================================
def estimate_reliability(
    question: MetaculusQuestion,
    research: str,
    reasoning_text: str,
    archetype: Archetype,
) -> float:
    """
    Heuristic r∈[0,1]. Adjusted by archetype (forecastability priors).
    """
    # Evidence strength: count distinct source lines in research (T/H/E)
    lines = [ln.strip() for ln in (research or "").splitlines() if ln.strip()]
    src_lines = [ln for ln in lines if re.match(r"^\[(T|H|E)\d+\]\s", ln)]
    n = len(src_lines)
    e = min(1.0, n / 6.0)
    
    # Resolution clarity
    rc = (getattr(question, "resolution_criteria", " ") or " ").lower()
    vague_markers = ["ambiguous", "subjective", "at discretion", "may decide", "likely", "reasonable"]
    objective_markers = ["will resolve", "according to", "published", "official", "announced", "measured", "reported"]
    c = 0.55
    if any(m in rc for m in objective_markers):
        c += 0.20
    if any(m in rc for m in vague_markers) or len(rc.strip()) < 40:
        c -= 0.20
    c = clamp01(c)

    # Time horizon
    now = datetime.now(timezone.utc)
    close_time = getattr(question, "close_time", None)
    if close_time:
        days = max(0.0, (close_time - now).total_seconds() / 86400.0)
        if days <= 30:
            h = 0.90
        elif days <= 180:
            h = 0.90 - (days - 30) * (0.30 / 150.0)
        elif days <= 365:
            h = 0.60 - (days - 180) * (0.20 / 185.0)
        elif days <= 730:
            h = 0.40 - (days - 365) * (0.15 / 365.0)
        else:
            h = 0.25
    else:
        h = 0.45

    # Base-rate anchoring: look for explicit outside-view language
    rt = (reasoning_text or " ").lower()
    b = 0.25
    if "base rate" in rt or "reference class" in rt or "outside view" in rt:
        b = 0.85
    elif "historical" in rt or "prior" in rt:
        b = 0.60

    # Archetype forecastability prior
    archetype_prior = {
        Archetype.MARKET: 0.70,
        Archetype.SPORTS_ENT: 0.65,
        Archetype.MACRO: 0.55,
        Archetype.ELECTION: 0.55,
        Archetype.SCI_TECH: 0.45,
        Archetype.ORG_PRODUCT: 0.45,
        Archetype.WEATHER_NATURAL: 0.50,
        Archetype.GEO_CONFLICT: 0.35,
        Archetype.OTHER: 0.40,
    }[archetype]

    r = 0.28 * e + 0.20 * c + 0.15 * h + 0.22 * b + 0.15 * archetype_prior

    # Penalize handwaving: lots of words, few numbers
    n_numbers = len(re.findall(r"\b\d+(\.\d+)?\b", reasoning_text or " "))
    if len((reasoning_text or " ")) > 350 and n_numbers < 4:
        r -= 0.10

    if any(w in rt for w in ["mixed", "unclear", "contradict", "conflicting", "hard to say"]):
        r -= 0.07

    # If archetype is GEO, apply extra caution unless criteria is very objective
    if archetype == Archetype.GEO_CONFLICT and c < 0.6:
        r -= 0.05

    return clamp01(r)


def extremize_binary(p_raw: float, r: float, tuning: ArchetypeTuning) -> float:
    """
    Reliability-driven log-odds extremization. Archetype tuning sets max k.
    Conservative but bold: only extremize when reliability is high.
    """
    p_raw = clamp01(p_raw)
    r = clamp01(r)
    if r < 0.35:
        k = 0.85  # De-extremize when unreliable
    elif r <= 0.65:
        k = 1.0   # No change
    else:
        # Scale to max_k
        frac = (r - 0.65) / 0.35
        k = 1.0 + (tuning.max_k_binary - 1.0) * frac

    return sigmoid(k * logit(p_raw))


def extremize_mc(probs: List[float], r: float, tuning: ArchetypeTuning) -> List[float]:
    """
    MC analogue: p'_i ∝ p_i^k, with k from reliability and archetype max_k_mc.
    """
    r = clamp01(r)
    if not probs:
        return probs
    if r < 0.35:
        k = 0.90
    elif r <= 0.65:
        k = 1.00
    else:
        frac = (r - 0.65) / 0.35
        k = 1.0 + (tuning.max_k_mc - 1.0) * frac

    ps = [max(0.0, float(p)) for p in probs]
    s = sum(ps)
    if s <= 0:
        return [1.0 / len(ps) for _ in ps]
    ps = [p / s for p in ps]

    ps2 = [p**k for p in ps]
    s2 = sum(ps2)
    if s2 <= 0:
        return [1.0 / len(ps) for _ in ps]
    return [p / s2 for p in ps2]


def adjust_percentiles_spread(pcts: List[Percentile], r: float, archetype: Archetype) -> List[Percentile]:
    """
    Numeric sharpness: low r -> widen tails; high r -> tighten.
    Archetype slightly adjusts tightening/widening behavior.
    """
    if not pcts:
        return pcts
    r = clamp01(r)
    # Archetype modifiers: market/macro distributions can be sharper; geo/other should be wider.
    arche_mult = {
        Archetype.MARKET: 0.90,
        Archetype.MACRO: 0.95,
        Archetype.SPORTS_ENT: 0.92,
        Archetype.ELECTION: 0.98,
        Archetype.SCI_TECH: 1.05,
        Archetype.ORG_PRODUCT: 1.05,
        Archetype.WEATHER_NATURAL: 1.00,
        Archetype.GEO_CONFLICT: 1.10,
        Archetype.OTHER: 1.05,
    }[archetype]

    s = (1.25 - 0.60 * r) * arche_mult
    s = max(0.55, min(1.70, s))

    pct_map = {float(p.percentile): float(p.value) for p in pcts}
    if 0.5 in pct_map:
        med = pct_map[0.5]
    elif 0.4 in pct_map and 0.6 in pct_map:
        med = 0.5 * (pct_map[0.4] + pct_map[0.6])
    else:
        med = sorted([float(p.value) for p in pcts])[len(pcts) // 2]

    adjusted: List[Percentile] = []
    for p in pcts:
        val = float(p.value)
        new_val = med + (val - med) * s
        adjusted.append(Percentile(percentile=float(p.percentile), value=new_val))

    return adjusted


# =========================================================
# 🤖 UpskillBot (Type-Aware, Conservative but Bold)
# =========================================================
class UpskillBot(ForecastBot):
    _max_concurrent_questions = MAX_CONCURRENT_QUESTIONS
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

        # Archetype caches (avoid repeated classification cost)
        self._archetype_cache: Dict[str, Archetype] = {}

        # Optional cost tracking (only active if tiktoken installed)
        self._cost_tracker: Dict[str, Dict[str, int]] = {}
        self._model_pricing = {
            "gpt-4.1-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
            "gpt-5.2": {"input": 2.50, "output": 10.00},
            "gpt-5.1": {"input": 2.00, "output": 8.00},
        }
        self._encoding_cache: Dict[str, Any] = {}

    def _llm_config_defaults(self) -> dict[str, str]:
        return {"default": DEFAULT_FORECASTER, "parser": PARSER_MODEL, "summarizer": SUMMARIZER_MODEL}

    def _get_encoding(self, model_name: str):
        try:
            import tiktoken
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

    async def detect_archetype(self, question: MetaculusQuestion) -> Archetype:
        """
        Rules first; LLM fallback if uncertain.
        Cached per question id/digest.
        """
        key = stable_question_cache_key(question)
        if key in self._archetype_cache:
            return self._archetype_cache[key]

        blob = _question_text_blob(question)
        guess, conf = rule_archetype(blob)
        if guess and conf >= 0.8:
            self._archetype_cache[key] = guess
            return guess

        # LLM fallback via parser model (cheap)
        prompt = clean_indents(f"""
        Classify this forecasting question into exactly one archetype:
        - market
        - election_politics
        - sci_tech
        - macro
        - geo_conflict
        - org_product
        - weather_natural
        - sports_entertainment
        - other

        Return ONLY the archetype string.

        QUESTION:
        {question.question_text}

        BACKGROUND:
        {question.background_info or "None"}

        RESOLUTION:
        {question.resolution_criteria or "None"}
        """)
        out = (await self._invoke("parser", prompt)).strip().lower()
        mapping = {a.value: a for a in Archetype}
        a = mapping.get(out, Archetype.OTHER)
        self._archetype_cache[key] = a
        return a

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Compatible with ForecastBot base class signature.
        Detects archetype internally.
        """
        archetype = await self.detect_archetype(question)
        cache_key = f"{stable_question_cache_key(question)}::{archetype.value}"
        if cache_key in self._research_cache:
            return self._research_cache[cache_key]

        tuning = archetype_tuning(archetype)

        async with self._concurrency_limiter:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            base_query = build_tavily_query(question)

            # Tavily recent
            try:
                recent_query = strict_truncate_query(base_query, tuning.recent_suffix, 395)
                recent = await self._tavily_search(
                    recent_query, search_depth="advanced", max_results=4, days=180
                )
                recent_snips = []
                for i, r in enumerate((recent.get("results", []) or [])[:4]):
                    title = r.get("title", " ") or " "
                    url = r.get("url", " ") or " "
                    content = r.get("content") or " "
                    snippet = textwrap.shorten(content, width=160, placeholder="…")
                    recent_snips.append(f"[T{i+1}] {title} ({url}): {snippet}")
                recent_summary = "\n".join(recent_snips) if recent_snips else "[T] No recent results"
            except Exception as e:
                logger.error(f"Recent Tavily failed: {e}")
                recent_summary = f"[T] Error: {e}"

            # Tavily historical/base rate
            try:
                historical_query = strict_truncate_query(base_query, tuning.historical_suffix, 395)
                historical = await self._tavily_search(
                    historical_query, search_depth="advanced", max_results=4
                )
                hist_snips = []
                for i, r in enumerate((historical.get("results", []) or [])[:4]):
                    title = r.get("title", " ") or " "
                    url = r.get("url", " ") or " "
                    content = r.get("content") or " "
                    snippet = textwrap.shorten(content, width=160, placeholder="…")
                    hist_snips.append(f"[H{i+1}] {title} ({url}): {snippet}")
                historical_summary = "\n".join(hist_snips) if hist_snips else "[H] No historical results"
            except Exception as e:
                logger.error(f"Historical Tavily failed: {e}")
                historical_summary = f"[H] Error: {e}"

            # Exa
            try:
                exa_query = strict_truncate_query(base_query, tuning.recent_suffix, 395)
                exa_results = await self.exa.search(exa_query, num_results=4) if self.exa else []
                if exa_results:
                    exa_snips = []
                    for i, r in enumerate(exa_results[:4]):
                        title = r.get("title", " ") or " "
                        url = r.get("url", " ") or " "
                        text = r.get("text") or r.get("snippet") or " "
                        snippet = textwrap.shorten(text, width=160, placeholder="…")
                        exa_snips.append(f"[E{i+1}] {title} ({url}): {snippet}")
                    exa_summary = "\n".join(exa_snips)
                else:
                    exa_summary = "[E] No Exa results (or EXA_API_KEY missing)"
            except Exception as e:
                logger.error(f"Exa search failed: {e}")
                exa_summary = f"[E] Error: {e}"

            research = clean_indents(f"""
            ### Research (as of {today_str}) — Archetype: {archetype.value}
            {recent_summary}

            {historical_summary}

            {exa_summary}
            """).strip()

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
                "title": (getattr(question, "question_text", "Unknown") or " ")[:120],
                "type": question.__class__.__name__,
                "predicted_prob": prob,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "difficulty_score": self._estimate_question_difficulty(question),
                "reasoning_snippet": shorten_reasoning(reasoning, 400).replace("\n", "  "),
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

    def _estimate_question_difficulty(self, question: MetaculusQuestion) -> float:
        text = ((question.question_text or " ") + "  " + (question.background_info or " ")).lower()
        now = datetime.now(timezone.utc)
        days_to_close = (
            (question.close_time - now).total_seconds() / 86400 if getattr(question, "close_time", None) else 365
        )
        base_rate_hint = any(w in text for w in ["rare", "unlikely", "first time", "never before", "unprecedented"])
        long_horizon = days_to_close > 365
        vague_resolution = "ambiguous" in (question.resolution_criteria or " ").lower()
        return min(1.0, 0.3 + 0.3 * long_horizon + 0.2 * base_rate_hint + 0.2 * vague_resolution)

    def _archetype_prompt_header(self, archetype: Archetype) -> str:
        # Small tailored advice that nudges the LLM into better domain-specific moves.
        if archetype == Archetype.MARKET:
            return "Treat markets as partially efficient: use consensus as base rate; focus on catalysts and timing."
        if archetype == Archetype.ELECTION:
            return "Use polling+fundamentals: base rate on incumbency/seat history; watch polling error and turnout."
        if archetype == Archetype.MACRO:
            return "Use consensus forecasts as priors; respect typical forecast errors; watch next release dates."
        if archetype == Archetype.GEO_CONFLICT:
            return "Be cautious: heavy-tailed uncertainty; prioritize actions over statements; use conflict base rates."
        if archetype == Archetype.WEATHER_NATURAL:
            return "Anchor strongly to historical frequencies and seasonality; only go extreme with official alerts."
        if archetype == Archetype.SCI_TECH:
            return "Use rate-of-progress reference class; penalize hype; require concrete milestones and timelines."
        if archetype == Archetype.ORG_PRODUCT:
            return "Use org cadence priors; weigh official signals and incentives; watch for execution risk."
        if archetype == Archetype.SPORTS_ENT:
            return "Use odds/ratings as priors; incorporate injuries/schedule; avoid overreacting to noisy recent games."
        return "Use outside view first; be bold only with strong evidence and clear resolution."

    async def _run_forecast_with_red_team(
        self, question: MetaculusQuestion, research: str, archetype: Archetype
    ) -> Tuple[str, float, float]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        header = self._archetype_prompt_header(archetype)
        tuning = archetype_tuning(archetype)

        initial_prompt = clean_indents(f"""
        You are a calibrated superforecaster. Today (UTC): {today}.
        ARCHETYPE: {archetype.value}. Guidance: {header}

        PROCESS (conservative, but not timid):
        - Start with outside view: explicitly state reference class + base rate (or range).
        - Then inside view: list key evidence with direction (+/–) and strength (1–5).
        - Update from base rate to a "raw" probability.
        - Be willing to go bold ONLY if evidence is strong and resolution is clear.

        WRITING:
        - Keep explanation short: 6–10 bullets total.
        - Include 2–3 signposts with conditional updates ("If X, move to ~Y%").

        FORMAT (must end with EXACT line "Probability: ZZ%"):
        Approach: <1 sentence>
        Base rate: <1 line, include a % or range>
        Evidence:
        - <bullet (≤1 line)>
        ...
        Signposts:
        - If <event>, update to ~<p>%
        - If <event>, update to ~<p>%
        Uncertainties:
        - <2 bullets>
        Final: <1 sentence>
        Probability: ZZ%

        QUESTION:
        {question.question_text}

        BACKGROUND:
        {question.background_info or 'None'}

        RESOLUTION CRITERIA:
        {question.resolution_criteria or 'None'}

        RESEARCH:
        {research}
        """)
        initial_text = await self._invoke("default", initial_prompt)

        red_team_prompt = clean_indents(f"""
        You are a skeptical reviewer for archetype={archetype.value}.
        Push back against overconfidence and weak reference classes.

        Provide:
        - 3 strongest counterarguments (bullets)
        - 2 resolution traps / ambiguity checks (bullets)
        - 2 signposts that would move probability substantially (bullets)
        - Calibration note: should we be closer to base rate / 50%?

        FORECAST:
        {initial_text}

        QUESTION:
        {question.question_text}

        RESEARCH:
        {research}
        """)
        critique = await self._invoke("default", red_team_prompt)

        final_prompt = clean_indents(f"""
        Revise the forecast using the critique. Keep archetype guidance in mind.
        Rules:
        - Keep outside view explicit (reference class + base rate).
        - Make the smallest necessary adjustment.
        - If critique reveals weak evidence or resolution ambiguity, move toward base rate/50%.
        - If critique is answered with strong evidence, you MAY go bolder.
        - Keep same format and end with: Probability: ZZ%

        ORIGINAL:
        {initial_text}

        CRITIQUE:
        {critique}
        """)
        revised_text = await self._invoke("default", final_prompt)

        # Parse raw probability
        p_raw: Optional[float] = None
        try:
            parsed: BinaryPrediction = await structure_output(
                revised_text, BinaryPrediction, model=self.get_llm("parser", "llm")
            )
            p_raw = float(parsed.prediction_in_decimal)
        except Exception:
            p_raw = extract_probability(revised_text)

        if p_raw is None:
            p_raw = 0.5

        # Reliability + type-aware extremization + type-aware clipping
        r = estimate_reliability(question, research, revised_text, archetype)
        p_final = extremize_binary(p_raw, r, tuning)
        p_final = conservative_clip(p_final, lo=tuning.clip_lo, hi=tuning.clip_hi)

        # Replace ONLY the last Probability line (avoid earlier matches)
        matches = list(re.finditer(r"Probability\s*:\s*[0-9]+(?:\.[0-9]+)?\s*%?\s*$",
                                  revised_text.strip(),
                                  flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            m = matches[-1]
            revised_text = revised_text[:m.start()] + f"Probability: {p_final*100:.1f}%" + revised_text[m.end():]
        else:
            revised_text = revised_text.strip() + f"\nProbability: {p_final*100:.1f}%"

        return shorten_reasoning(revised_text, 900), p_final, r

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        archetype = await self.detect_archetype(question)
        reasoning, prob, r = await self._run_forecast_with_red_team(question, research, archetype)
        
        # Build detailed reasoning with extremization audit trail
        tuning = archetype_tuning(archetype)
        detailed_reasoning = self._build_detailed_reasoning(
            question=question,
            reasoning=reasoning,
            archetype=archetype,
            reliability=r,
            prediction=prob,
            prediction_type="binary",
            tuning=tuning,
        )
        
        self._record_prediction(
            question,
            prob,
            detailed_reasoning,
            extra={"archetype": archetype.value, "reliability_r": round(float(r), 4)},
        )
        self._log_calibration(question, prob, detailed_reasoning, archetype, r)
        return ReasonedPrediction(prediction_value=prob, reasoning=detailed_reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[NormalizedPredictedOptionList]:
        archetype = await self.detect_archetype(question)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        header = self._archetype_prompt_header(archetype)
        tuning = archetype_tuning(archetype)

        prompt = clean_indents(f"""
        You are a calibrated superforecaster. Today (UTC): {today}.
        ARCHETYPE: {archetype.value}. Guidance: {header}

        PROCESS:
        - Start with outside view/reference class.
        - Then inside view: key evidence, short bullets.
        - Produce a "raw" probability allocation.
        - Be willing to concentrate probability ONLY if evidence is strong and resolution is clear.

        REQUIREMENTS:
        - Probabilities must be decimals in [0,1] and sum to 1.
        - Use option_name field with EXACT option text as provided.
        - Keep reasoning ≤ 8 bullets.
        - Include 2 signposts.

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
        Base rate: <1 line (if applicable)>
        Evidence: <≤8 bullets>
        Signposts:
        - If <event>, reallocate toward <option>
        - If <event>, reallocate toward <option>
        Then list options as:
        <Option text>: 0.23
        """)
        reasoning = await self._invoke("default", prompt)

        try:
            pred: NormalizedPredictedOptionList = await structure_output(
                reasoning,
                NormalizedPredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=(
                    f"Options: {question.options}. "
                    "Probabilities must be decimals (0-1) summing to 1. "
                    "Use 'option_name' field with EXACT option text. "
                ),
            )
            # Normalize any stray field names for robustness
            normalized_options = [_normalize_predicted_option(o) if isinstance(o, dict) else o for o in pred.predicted_options]
            pred = NormalizedPredictedOptionList(predicted_options=normalized_options)
        except Exception as e:
            logger.warning(f"MC parse failed Q{getattr(question, 'id', 'unknown')}: {e}")
            p = 1.0 / max(1, len(question.options))
            pred = NormalizedPredictedOptionList(
                predicted_options=[PredictedOption(option_name=opt, probability=p) for opt in question.options]
            )

        pred = map_to_valid_options(pred, question.options)

        r = estimate_reliability(question, research, reasoning, archetype)
        probs = [float(o.probability) for o in pred.predicted_options]
        probs2 = extremize_mc(probs, r, tuning)

        pred2 = NormalizedPredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=pred.predicted_options[i].option_name, probability=float(probs2[i]))
                for i in range(len(pred.predicted_options))
            ]
        )

        prob_dict = {opt.option_name: float(opt.probability) for opt in pred2.predicted_options}
        top_opt = max(prob_dict, key=prob_dict.get) if prob_dict else "N/A"
        top_prob = float(prob_dict[top_opt]) if prob_dict else None

        # Build detailed reasoning with extremization audit trail
        detailed_reasoning = self._build_detailed_reasoning(
            question=question,
            reasoning=shorten_reasoning(reasoning, 900),
            archetype=archetype,
            reliability=r,
            prediction=prob_dict,
            prediction_type="multiple_choice",
            tuning=tuning,
        )

        self._record_prediction(
            question,
            top_prob,
            detailed_reasoning,
            extra={"archetype": archetype.value, "top_option": top_opt, "reliability_r": round(float(r), 4)},
        )
        self._log_calibration(question, prob_dict, detailed_reasoning, archetype, r)
        return ReasonedPrediction(prediction_value=pred2, reasoning=detailed_reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        archetype = await self.detect_archetype(question)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        header = self._archetype_prompt_header(archetype)

        lower_msg = (
            f"Lower bound: {question.lower_bound}. " if question.lower_bound is not None else "No explicit lower bound. "
        )
        upper_msg = (
            f"Upper bound: {question.upper_bound}. " if question.upper_bound is not None else "No explicit upper bound. "
        )

        prompt = clean_indents(f"""
        You are a calibrated quantitative forecaster. Today (UTC): {today}.
        ARCHETYPE: {archetype.value}. Guidance: {header}

        PROCESS:
        - Start with outside view/reference class.
        - Then inside view: key evidence, short bullets.
        - Produce a distribution: widen tails if uncertain; tighten if reliable (but don't be unrealistically tight).

        REQUIREMENTS:
        - Respect bounds if given.
        - Keep reasoning ≤ 8 bullets.
        - Output MUST include a parsable percentile list.
        - Return VALID JSON only for the percentile list (no markdown).

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
        Base rate: <1 line (if possible)>
        Evidence: <≤8 bullets>
        Signposts:
        - If <event>, shift median to ~<value>
        - If <event>, widen/narrow tails
        PercentilesJSON:
        [
          {{"percentile":0.1,"value":123}},
          {{"percentile":0.2,"value":...}},
          {{"percentile":0.4,"value":...}},
          {{"percentile":0.6,"value":...}},
          {{"percentile":0.8,"value":...}},
          {{"percentile":0.9,"value":...}}
        ]
        """)
        reasoning = await self._invoke("default", prompt)

        r = estimate_reliability(question, research, reasoning, archetype)

        try:
            pct_list: list[Percentile] = await structure_output(
                reasoning, list[Percentile], model=self.get_llm("parser", "llm")
            )
            pct_list = adjust_percentiles_spread(pct_list, r, archetype)
            dist = NumericDistribution.from_question(pct_list, question)
        except Exception as e:
            logger.warning(f"Numeric parse failed: {e}")
            lo = float(question.lower_bound if question.lower_bound is not None else 0.0)
            hi = float(question.upper_bound if question.upper_bound is not None else lo + 1.0)
            fallback_ps = [0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
            fallback = [Percentile(percentile=p, value=lo + (hi - lo) * p) for p in fallback_ps]
            fallback = adjust_percentiles_spread(fallback, r, archetype)
            dist = NumericDistribution.from_question(fallback, question)

        # Build detailed reasoning with extremization audit trail
        tuning = archetype_tuning(archetype)
        detailed_reasoning = self._build_detailed_reasoning(
            question=question,
            reasoning=shorten_reasoning(reasoning, 900),
            archetype=archetype,
            reliability=r,
            prediction=[{"percentile": p.percentile, "value": p.value} for p in pct_list] if 'pct_list' in locals() else "fallback",
            prediction_type="numeric",
            tuning=tuning,
        )

        self._record_prediction(
            question,
            None,
            detailed_reasoning,
            extra={"archetype": archetype.value, "reliability_r": round(float(r), 4)},
        )
        self._log_calibration(question, dist, detailed_reasoning, archetype, r)
        return ReasonedPrediction(prediction_value=dist, reasoning=detailed_reasoning)

    def _build_detailed_reasoning(
        self,
        question: MetaculusQuestion,
        reasoning: str,
        archetype: Archetype,
        reliability: float,
        prediction: Any,
        prediction_type: str,
        tuning: ArchetypeTuning,
    ) -> str:
        """
        Build detailed, auditable reasoning with extremization audit trail.
        Matches Tude's multi-section format while preserving UpskillBot's archetype system.
        """
        qid = getattr(question, "id", "unknown")
        qtxt = getattr(question, "question_text", "")[:200]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Determine extremization details
        if reliability < 0.35:
            extremize_status = "De-extremized (low reliability)"
            extremize_reason = f"r={reliability:.2f} < 0.35 → k=0.85 (conservative)"
        elif reliability <= 0.65:
            extremize_status = "No extremization"
            extremize_reason = f"r={reliability:.2f} in [0.35, 0.65] → k=1.0 (neutral)"
        else:
            if prediction_type == "binary":
                k = 1.0 + (tuning.max_k_binary - 1.0) * ((reliability - 0.65) / 0.35)
            else:
                k = 1.0 + (tuning.max_k_mc - 1.0) * ((reliability - 0.65) / 0.35)
            extremize_status = f"Extremized (k={k:.2f})"
            extremize_reason = f"r={reliability:.2f} > 0.65 → bold update justified"

        # Format prediction value
        if prediction_type == "binary":
            pred_str = f"{prediction*100:.1f}%"
        elif prediction_type == "multiple_choice":
            top_opt = max(prediction, key=prediction.get) if isinstance(prediction, dict) else "N/A"
            top_prob = prediction.get(top_opt, 0) if isinstance(prediction, dict) else 0
            pred_str = f"{top_opt}: {top_prob*100:.1f}%"
        else:
            pred_str = "Distribution (see percentiles)"

        return clean_indents(f"""
        ## Forecast (Q{qid})
        **Date (UTC):** {today}
        **Archetype:** {archetype.value}
        **Question:** {qtxt}

        **Prediction:** {pred_str}

        ### 🔍 Evidence Synthesis
        {reasoning}

        ### 📊 Reliability Assessment
        - **Evidence strength:** {reliability:.2f}/1.00
        - **Extremization:** {extremize_status}
        - **Reason:** {extremize_reason}
        - **Archetype bounds:** [{tuning.clip_lo:.3f}, {tuning.clip_hi:.3f}]

        ### 🎯 Why This Prediction
        - **Archetype guidance:** {self._archetype_prompt_header(archetype)}
        - **Base rate:** Outside view applied per {archetype.value} reference class
        - **Inside view:** Evidence-weighted update from research (Tavily + Exa)
        - **Conservative-but-bold:** De-extremize when r<0.35, extremize when r>0.65

        ### ⚠️ Key Signposts
        - Watch for archetype-specific catalysts (see Evidence section)
        - Resolution criteria clarity: {'High' if reliability > 0.6 else 'Medium' if reliability > 0.35 else 'Low'}
        - Model confidence: {'High' if reliability > 0.65 else 'Medium' if reliability > 0.35 else 'Low'}

        ### 📈 Extremization Audit Trail
        - Raw probability: Model output (see Evidence)
        - Reliability score: {reliability:.4f}
        - Extremization factor k: {extremize_status.split('(')[-1].rstrip(')') if '(' in extremize_status else 'N/A'}
        - Final probability: {pred_str}
        - Clipping applied: [{tuning.clip_lo:.3f}, {tuning.clip_hi:.3f}]
        """).strip()

    def _log_calibration(
        self,
        question: MetaculusQuestion,
        prediction_value: Any,
        reasoning: str,
        archetype: Archetype,
        reliability: float,
    ):
        """Log forecast to JSONL for post-hoc calibration analysis (matches Tude's format)."""
        qid = getattr(question, "id", "unknown")
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question_id": qid,
            "question_type": question.__class__.__name__,
            "question_text": getattr(question, "question_text", "")[:500],
            "resolution_date": getattr(question, "resolution_date", None),
            "community_prediction": getattr(question, "community_prediction", None),
            "prediction_value": prediction_value,
            "archetype": archetype.value,
            "reliability_score": round(reliability, 4),
            "models_used": [DEFAULT_FORECASTER, PARSER_MODEL],
            "research_used": True,
            "searchers_used": ["tavily", "exa"] if os.getenv("EXA_API_KEY") else ["tavily"],
            "reasoning_snippet": reasoning[:1500],
        }
        try:
            with open(CALIBRATION_LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log calibration  {e}")

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

    def export_predictions_to_csv(self, filepath: str = PREDICTIONS_CSV_FILE):
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

    def export_cost_report(self, filepath: str = COSTS_CSV_FILE):
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

    def log_report_summary(self, reports: List[Any]):
        """Match Tude's report summary format."""
        successful = sum(1 for r in reports if not isinstance(r, Exception))
        failed = len(reports) - successful
        logger.info(f"----------------------------------------------------------------------------------------------------")
        logger.info(f"Bot: UpskillBot")
        if failed > 0:
            logger.info(f"❌ Exceptions: {failed}/{len(reports)} questions failed")
        logger.info(f"✅ Successful forecasts: {successful}/{len(reports)}")
        logger.info(f"📊 Calibration logs: {CALIBRATION_LOG_FILE}")
        logger.info(f"📊 Predictions CSV: {PREDICTIONS_CSV_FILE}")
        logger.info(f"💰 Cost report: {COSTS_CSV_FILE}")
        logger.info(f"----------------------------------------------------------------------------------------------------")


# =========================================================
# 🚀 Main Entry Point
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run UpskillBot.")
    parser.add_argument("--tournament-ids", nargs="+", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    client = MetaculusClient()
    default_ids = list(dict.fromkeys([
        "32916",
        "minibench",
        "market-pulse-26q1",
        getattr(client, "CURRENT_MINIBENCH_ID", "minibench"),
    ]))
    tournament_ids = args.tournament_ids or default_ids

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

    try:
        all_reports = []
        for tid in tournament_ids:
            logger.info(f"▶ Forecasting tournament: {tid}")

            for attempt in range(RETRY_MAX):
                try:
                    reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
                    all_reports.extend(reports)
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if "too many requests" in msg or "cloudflare" in msg or "1015" in msg or "429" in msg:
                        logger.error(f"Rate-limited on tournament {tid} (attempt {attempt+1}/{RETRY_MAX}): {e}")
                        base = min(RETRY_MAX_S, RETRY_BASE_S * (2 ** attempt))
                        jitter = random.uniform(0.0, base * 0.25)
                        time.sleep(base + jitter)
                        continue
                    raise

            time.sleep(float(os.getenv("TOURNAMENT_SLEEP_S", "8.0")))

        bot.log_report_summary(all_reports)
        asyncio.run(bot._compute_brier_scores())
        bot.export_predictions_to_csv()
        bot.export_cost_report()
        logger.info(f"🏁 UpskillBot run completed. Calibration logs saved to {CALIBRATION_LOG_FILE}")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
