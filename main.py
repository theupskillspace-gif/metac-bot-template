
import argparse
import asyncio
import json
import logging
import math
import os
import random
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.request import Request, urlopen

import dotenv

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DatePercentile,
    DateQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictionAffirmed,
    PredictionTypes,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
_CLAUDE_OPUS_MODEL   = "openrouter/anthropic/claude-opus-4-6"
_CLAUDE_SONNET_MODEL = "openrouter/anthropic/claude-sonnet-4-6"
_GPT_MODEL           = "openrouter/openai/gpt-5.4-mini"
DOMAINS = [
    "geopolitics", "economics", "technology", "science",
    "public_health", "environment", "sports", "finance", "social", "other",
]
GEO_SCOPES = ["global", "regional", "national", "local"]


@dataclass
class QuestionProfile:
    """Metadata inferred about a question before forecasting begins."""
    domain: str                    = "other"
    geo_scope: str                 = "global"
    geography: str                 = ""
    time_horizon_days: int         = 365
    is_quantitative: bool          = False
    confidence_in_profile: float   = 0.0


class QuestionAnalyser:
    """
    Uses an LLM to classify a question's domain, geography, and time horizon.
    Results inform ModellingStrategy and source selection.
    """

    def __init__(self, llm: GeneralLlm):
        self._llm = llm

    async def classify(self, question: MetaculusQuestion) -> QuestionProfile:
        prompt = clean_indents(
            f"""
            Classify the following forecasting question. Reply ONLY with a JSON
            object matching this exact schema (no markdown, no extra keys):

            {{
              "domain": "<one of: {', '.join(DOMAINS)}>",
              "geo_scope": "<one of: {', '.join(GEO_SCOPES)}>",
              "geography": "<country/region name, or empty string if global>",
              "time_horizon_days": <integer, estimated days until resolution>,
              "is_quantitative": <true if the answer is a number or date, false otherwise>,
              "confidence_in_profile": <float 0.0-1.0>
            }}

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            """
        )
        try:
            raw = await self._llm.invoke(prompt)
            raw = raw.strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
            data = json.loads(raw)
            return QuestionProfile(
                domain=data.get("domain", "other"),
                geo_scope=data.get("geo_scope", "global"),
                geography=data.get("geography", ""),
                time_horizon_days=int(data.get("time_horizon_days", 365)),
                is_quantitative=bool(data.get("is_quantitative", False)),
                confidence_in_profile=float(data.get("confidence_in_profile", 0.5)),
            )
        except Exception as exc:
            logger.warning(f"[Analyser] Failed to classify question: {exc}")
            return QuestionProfile()


# ===========================================================================
# 2. MODELLING STRATEGY
# ===========================================================================

class ModellingStrategy:
    """
    Strategies
    ----------
    base_rate      – anchor on historical frequencies of similar events
    trend          – extrapolate a measurable trend forward in time
    analogical     – reason from a close historical analogy
    market_signal  – weight prediction-market / expert-survey signals heavily
    """

    @staticmethod
    def select(profile: QuestionProfile) -> str:
        if profile.domain in ("economics", "finance") and profile.is_quantitative:
            return "trend"
        if profile.domain in ("geopolitics", "social"):
            return "analogical"
        if profile.time_horizon_days < 60:
            return "market_signal"
        return "base_rate"

    @staticmethod
    def get_prompt_block(strategy: str, profile: QuestionProfile) -> str:
        geo_ctx = f" focusing on {profile.geography}" if profile.geography else ""

        if strategy == "trend":
            return clean_indents(
                f"""
                ## Strategy: Trend Extrapolation{geo_ctx}
                1. Identify the key measurable variable.
                2. Find its recent trajectory (last 1-3 data points).
                3. Project forward to resolution date.
                4. Apply mean-reversion: trends rarely persist at full strength.
                5. Bound estimate with a realistic uncertainty range.
                """
            ).strip()

        if strategy == "analogical":
            return clean_indents(
                f"""
                ## Strategy: Analogical Reasoning{geo_ctx}
                1. Identify 2-3 structurally similar historical situations.
                2. How did those resolve? What was the base rate?
                3. Key SIMILARITIES – how they support your estimate.
                4. Key DIFFERENCES – how they require adjustment.
                5. Weight analogies by structural similarity, not surface resemblance.
                """
            ).strip()

        if strategy == "market_signal":
            return clean_indents(
                f"""
                ## Strategy: Market Signal{geo_ctx}
                1. Check prediction markets (Metaculus, Polymarket, Metaforecast).
                2. If a signal exists, treat it as a strong prior.
                3. Adjust only if you have concrete information it hasn't priced in.
                4. Short horizons: weight inertia very heavily.
                """
            ).strip()

        # default: base_rate
        return clean_indents(
            f"""
            ## Strategy: Base Rate{geo_ctx}
            1. Define the reference class for this type of event.
            2. Historical frequency of the outcome in that class.
            3. Anchor to that base rate.
            4. Apply inside-view adjustments only for clear distinguishing features.
            5. Limit total adjustment from base rate to ±20 pp unless evidence is overwhelming.
            """
        ).strip()


# ===========================================================================
# 3. PLUGGABLE SOURCE REGISTRY
# ===========================================================================

class BaseSource(ABC):
    """Abstract base class for any information source."""
    name: str = "unnamed_source"

    @abstractmethod
    async def fetch(self, query: str) -> str:
        ...

    def is_available(self) -> bool:
        return True


class SourceRegistry:
    """Holds all registered information sources."""

    def __init__(self):
        self._sources: list[BaseSource] = []

    def register(self, source: BaseSource) -> None:
        self._sources.append(source)
        logger.info(f"[SourceRegistry] Registered source: {source.name}")

    def available_sources(self) -> list[BaseSource]:
        return [s for s in self._sources if s.is_available()]

    async def fetch_all(self, query: str) -> list[str]:
        sources = self.available_sources()
        tasks   = [s.fetch(query) for s in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        blocks: list[str] = []
        for src, res in zip(sources, results):
            if isinstance(res, Exception):
                blocks.append(f"[{src.name}] Query failed: {res}")
            elif isinstance(res, str) and res.strip():
                blocks.append(f"[{src.name}]\n{res}")
        return blocks


# ---------------------------------------------------------------------------
# TavilySource
# ---------------------------------------------------------------------------

def _format_tavily_results(query: str, results: dict[str, Any], max_results: int = 6) -> str:
    items = results.get("results", []) or []
    lines = [f"Query: {query}"]
    for r in items[:max_results]:
        title   = (r.get("title")       or "").strip()
        url     = (r.get("url")         or "").strip()
        snippet = (r.get("content")     or "").strip()
        raw     = (r.get("raw_content") or "").strip()
        if title or url or snippet:
            lines.append(f"- {title}")
            if url:
                lines.append(f"  URL: {url}")
            if snippet:
                lines.append(f"  Notes: {snippet}")
            if raw and raw != snippet:
                lines.append(f"  Full text (truncated): {raw[:1500]}")
    return "\n".join(lines).strip()


class TavilySearcher:
    def __init__(
        self,
        api_key: str,
        max_results: int = 6,
        search_depth: str = "advanced",
        include_answer: bool = False,
        include_raw_content: bool = True,
        include_images: bool = False,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        timeout_s: int = 30,
    ):
        self.api_key             = api_key
        self.max_results         = max_results
        self.search_depth        = search_depth
        self.include_answer      = include_answer
        self.include_raw_content = include_raw_content
        self.include_images      = include_images
        self.include_domains     = include_domains
        self.exclude_domains     = exclude_domains
        self.timeout_s           = timeout_s

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req  = Request(url, data=data,
                       headers={"Content-Type": "application/json",
                                "Accept": "application/json"},
                       method="POST")
        with urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    async def search(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "api_key": self.api_key, "query": query,
            "max_results": self.max_results,
            "search_depth": self.search_depth,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
            "include_images": self.include_images,
        }
        if self.include_domains:  payload["include_domains"]  = self.include_domains
        if self.exclude_domains:  payload["exclude_domains"]  = self.exclude_domains
        return await asyncio.to_thread(self._post_json, "https://api.tavily.com/search", payload)


class TavilySource(BaseSource):
    """Wraps TavilySearcher as a pluggable source."""
    name = "tavily_web"

    def __init__(self, api_key: str, include_domains: list[str] | None = None,
                 exclude_domains: list[str] | None = None):
        self._api_key  = api_key
        self._searcher = TavilySearcher(
            api_key=api_key,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        ) if api_key else None

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def fetch(self, query: str) -> str:
        if not self._searcher:
            return ""
        try:
            results = await self._searcher.search(query)
            return _format_tavily_results(query, results, self._searcher.max_results)
        except Exception as exc:
            return f"Query: {query}\n- Tavily failed: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# ExaSource  (neural semantic search)
# ---------------------------------------------------------------------------

class ExaSource(BaseSource):
    """
    Exa neural search (https://exa.ai).
    Requires EXA_API_KEY in environment.
    Docs: https://docs.exa.ai/reference/search
    """
    name = "exa_neural"
    _API_URL = "https://api.exa.ai/search"

    def __init__(self, api_key: str, num_results: int = 5, use_autoprompt: bool = True,
                 timeout_s: int = 30):
        self._api_key      = api_key
        self._num_results  = num_results
        self._autoprompt   = use_autoprompt
        self._timeout_s    = timeout_s

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req  = Request(
            self._API_URL, data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": self._api_key,
            },
            method="POST",
        )
        with urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    async def fetch(self, query: str) -> str:
        if not self._api_key:
            return ""
        try:
            payload = {
                "query": query,
                "numResults": self._num_results,
                "useAutoprompt": self._autoprompt,
                "contents": {"text": {"maxCharacters": 1500}},
            }
            raw = await asyncio.to_thread(self._post_json, payload)
            results = raw.get("results", [])
            lines = [f"Query: {query}"]
            for r in results:
                title   = (r.get("title")   or "").strip()
                url     = (r.get("url")     or "").strip()
                excerpt = (r.get("text")    or "").strip()
                score   = r.get("score", 0.0)
                lines.append(f"- {title}  [score={score:.3f}]")
                if url:
                    lines.append(f"  URL: {url}")
                if excerpt:
                    lines.append(f"  Excerpt: {excerpt[:1200]}")
            return "\n".join(lines).strip()
        except Exception as exc:
            return f"Query: {query}\n- Exa failed: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# AskNewsSource  (real-time news with AI summaries)
# ---------------------------------------------------------------------------

class AskNewsSource(BaseSource):
    """
    AskNews API (https://asknews.app).
    Requires ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET in environment.
    Uses OAuth2 client-credentials to get a bearer token, then queries /v1/news/search.
    """
    name = "asknews_news"
    _TOKEN_URL  = "https://auth.asknews.app/oauth2/token"
    _SEARCH_URL = "https://api.asknews.app/v1/news/search"

    def __init__(self, client_id: str, client_secret: str,
                 n_articles: int = 6, timeout_s: int = 30):
        self._client_id     = client_id
        self._client_secret = client_secret
        self._n_articles    = n_articles
        self._timeout_s     = timeout_s
        self._token: str    = ""
        self._token_expiry: float = 0.0

    def is_available(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def _get_token_sync(self) -> str:
        """Fetch (or reuse) an OAuth2 bearer token."""
        if time.time() < self._token_expiry - 60:
            return self._token
        payload = (
            f"grant_type=client_credentials"
            f"&client_id={self._client_id}"
            f"&client_secret={self._client_secret}"
        ).encode("utf-8")
        req = Request(
            self._TOKEN_URL, data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        self._token        = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600)
        return self._token

    def _search_sync(self, query: str) -> dict[str, Any]:
        token  = self._get_token_sync()
        params = f"query={query.replace(' ', '+')}&n_articles={self._n_articles}&return_type=both"
        url    = f"{self._SEARCH_URL}?{params}"
        req    = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
        with urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    async def fetch(self, query: str) -> str:
        if not self.is_available():
            return ""
        try:
            raw = await asyncio.to_thread(self._search_sync, query)
            articles = raw.get("articles", []) or raw.get("data", {}).get("articles", [])
            lines = [f"Query: {query}"]
            for art in articles[: self._n_articles]:
                title   = (art.get("eng_title") or art.get("title") or "").strip()
                summary = (art.get("summary")   or "").strip()
                url     = (art.get("article_url") or art.get("url") or "").strip()
                pub_at  = (art.get("pub_date")  or "").strip()
                lines.append(f"- [{pub_at}] {title}")
                if url:
                    lines.append(f"  URL: {url}")
                if summary:
                    lines.append(f"  Summary: {summary[:1000]}")
            return "\n".join(lines).strip()
        except Exception as exc:
            return f"Query: {query}\n- AskNews failed: {type(exc).__name__}: {exc}"


# ===========================================================================
# 4. FORECAST VALIDATOR
# ===========================================================================

@dataclass
class ValidationRecord:
    question_url:          str
    question_text:         str
    domain:                str
    geo_scope:             str
    strategy:              str
    prediction_value:      str
    confidence_score:      float
    flagged_low_confidence: bool
    ts: float = field(default_factory=time.time)


class ForecastValidator:
    """
    Tracks every forecast, computes a heuristic confidence score, and persists
    a ledger to SQLite for post-hoc calibration analysis.
    """

    LOW_CONFIDENCE_THRESHOLD = 0.35

    def __init__(self, db_path: str = "upskillbot_validation.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_ledger (
                    question_url     TEXT,
                    question_text    TEXT,
                    domain           TEXT,
                    geo_scope        TEXT,
                    strategy         TEXT,
                    prediction_value TEXT,
                    confidence_score REAL,
                    flagged          INTEGER,
                    ts               REAL
                )
                """
            )
            conn.commit()

    def compute_confidence(
        self,
        prediction_value: Any,
        profile: QuestionProfile,
        research_length: int,
    ) -> float:
        classifier_score = profile.confidence_in_profile
        evidence_score   = min(1.0, research_length / 3000)
        if isinstance(prediction_value, float):
            signal_score = abs(prediction_value - 0.5) * 2
        else:
            signal_score = 0.5
        score = 0.40 * classifier_score + 0.35 * evidence_score + 0.25 * signal_score
        return round(min(1.0, max(0.0, score)), 3)

    def validate(
        self,
        question: MetaculusQuestion,
        profile: QuestionProfile,
        strategy: str,
        prediction_value: Any,
        research: str,
    ) -> ValidationRecord:
        confidence = self.compute_confidence(prediction_value, profile, len(research))
        flagged    = confidence < self.LOW_CONFIDENCE_THRESHOLD
        record = ValidationRecord(
            question_url=question.page_url,
            question_text=question.question_text[:300],
            domain=profile.domain,
            geo_scope=profile.geo_scope,
            strategy=strategy,
            prediction_value=str(prediction_value)[:200],
            confidence_score=confidence,
            flagged_low_confidence=flagged,
        )
        self._persist(record)
        level = logging.WARNING if flagged else logging.INFO
        logger.log(
            level,
            f"[Validator] confidence={confidence:.2f} flagged={flagged} "
            f"domain={profile.domain} strategy={strategy} | {question.page_url}",
        )
        return record

    def _persist(self, record: ValidationRecord) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO forecast_ledger
                    (question_url, question_text, domain, geo_scope, strategy,
                     prediction_value, confidence_score, flagged, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (record.question_url, record.question_text, record.domain,
                     record.geo_scope, record.strategy, record.prediction_value,
                     record.confidence_score, int(record.flagged_low_confidence), record.ts),
                )
                conn.commit()
        except Exception as exc:
            logger.warning(f"[Validator] Persist failed: {exc}")

    def summary(self) -> dict[str, Any]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT domain, COUNT(*) as n,
                           AVG(confidence_score) as avg_conf,
                           SUM(flagged) as n_flagged
                    FROM forecast_ledger GROUP BY domain ORDER BY n DESC
                    """
                ).fetchall()
            return {
                "by_domain": [
                    {"domain": r[0], "n": r[1],
                     "avg_confidence": round(r[2], 3), "n_flagged": r[3]}
                    for r in rows
                ]
            }
        except Exception:
            return {}


# ===========================================================================
# 5. CLIENT SPECIALISATION
# ===========================================================================

@dataclass
class ClientSpecialisation:
    """
    Optional configuration block for client-specific tuning.
    Inject at UpskillBot construction time.
    """
    domain_focus:       list[str] = field(default_factory=list)
    trusted_domains:    list[str] = field(default_factory=list)
    excluded_domains:   list[str] = field(default_factory=list)
    extra_context:      str       = ""
    calibration_target: float     = 0.15


# ===========================================================================
# 6. PERSISTENT RESEARCH CACHE (SQLite)
# ===========================================================================

class ResearchCache:
    def __init__(self, db_path: str = "upskillbot_cache.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_cache (
                    url TEXT PRIMARY KEY, content TEXT NOT NULL, ts REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _get_sync(self, url: str) -> str | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT content FROM research_cache WHERE url = ?", (url,)
            ).fetchone()
        return row[0] if row else None

    def _set_sync(self, url: str, content: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_cache (url, content, ts) VALUES (?, ?, ?)",
                (url, content, time.time()),
            )
            conn.commit()

    async def get(self, url: str) -> str | None:
        return await asyncio.to_thread(self._get_sync, url)

    async def set(self, url: str, content: str) -> None:
        await asyncio.to_thread(self._set_sync, url, content)


# ===========================================================================
# 7. EXTREMIZATION HELPERS
#    Aggressive logit-space extremization (factor=1.65) + conservativeness gate.
# ===========================================================================

@dataclass
class ExtremizationConfig:
    enabled: bool  = True
    factor:  float = 1.65   # raised from 1.45 → more extreme
    floor:   float = 0.02
    ceil:    float = 0.98
    # Conservativeness gate: hard-clip extremes before publishing
    conservative_floor: float = 0.03
    conservative_ceil:  float = 0.97


def _logit(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1.0 / (1.0 + z)
    z = math.exp(x); return z / (1.0 + z)


def extremize_probability(p: float, cfg: ExtremizationConfig) -> float:
    """Extremize then clip with conservativeness gate."""
    if not cfg.enabled:
        extremized = max(cfg.floor, min(cfg.ceil, p))
    else:
        extremized = max(cfg.floor, min(cfg.ceil, _sigmoid(_logit(p) * cfg.factor)))
    # Conservativeness gate: never publish outside [conservative_floor, conservative_ceil]
    return max(cfg.conservative_floor, min(cfg.conservative_ceil, extremized))


# ===========================================================================
# 8. MAIN BOT CLASS
# ===========================================================================

class UpskillBot(ForecastBot):
    """
    UpskillBot – superforecaster bot with multi-API research (Exa, Tavily, AskNews),
    aggressive extremization, and a conservativeness gate before publishing.
    """

    _max_concurrent_questions = 3
    _concurrency_limiter      = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    _min_seconds_between_search_calls = 1.2
    _min_seconds_between_llm_calls    = 0.35
    _last_search_call_ts = 0.0
    _last_llm_call_ts    = 0.0

    def __init__(self, *args, client_spec: ClientSpecialisation | None = None, **kwargs):
        llms = kwargs.pop("llms", None)
        if llms is None:
            opus_llm   = GeneralLlm(model=_CLAUDE_OPUS_MODEL,   temperature=0.15, timeout=90, allowed_tries=3)
            sonnet_llm = GeneralLlm(model=_CLAUDE_SONNET_MODEL, temperature=0.15, timeout=60, allowed_tries=3)
            gpt_llm    = GeneralLlm(model=_GPT_MODEL,           temperature=0.15, timeout=60, allowed_tries=3)
            llms = {
                "default":    opus_llm,
                "summarizer": sonnet_llm,
                "researcher": gpt_llm,
                "parser":     gpt_llm,
            }
        super().__init__(*args, llms=llms, **kwargs)

        self._client_spec    = client_spec or ClientSpecialisation()
        self._research_cache = ResearchCache()
        self._validator      = ForecastValidator()
        self._analyser       = QuestionAnalyser(self.get_llm("researcher", "llm"))

        # Source registry: Exa + Tavily + AskNews
        self._sources = SourceRegistry()

        exa_key = os.getenv("EXA_API_KEY", "").strip()
        self._sources.register(ExaSource(api_key=exa_key))

        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        self._sources.register(TavilySource(
            api_key=tavily_key,
            include_domains=self._client_spec.trusted_domains or None,
            exclude_domains=self._client_spec.excluded_domains or None,
        ))

        asknews_id     = os.getenv("ASKNEWS_CLIENT_ID",     "").strip()
        asknews_secret = os.getenv("ASKNEWS_CLIENT_SECRET", "").strip()
        self._sources.register(AskNewsSource(
            client_id=asknews_id,
            client_secret=asknews_secret,
        ))

        self._ext_cfg = ExtremizationConfig(
            enabled=os.getenv("EXTREMIZE_ENABLED", "true").lower() in ["1","true","yes","y"],
            factor=float(os.getenv("EXTREMIZE_FACTOR", "1.65")),
            floor=float(os.getenv("EXTREMIZE_FLOOR",  "0.02")),
            ceil=float(os.getenv("EXTREMIZE_CEIL",    "0.98")),
            conservative_floor=float(os.getenv("CONSERVATIVE_FLOOR", "0.03")),
            conservative_ceil=float(os.getenv("CONSERVATIVE_CEIL",  "0.97")),
        )

    # -------------------------------------------------------------------
    # Public API: register additional client data sources
    # -------------------------------------------------------------------

    def register_source(self, source: BaseSource) -> None:
        """
        Register a custom data source (proprietary DB, specialised API, etc.).

        Example
        -------
        class MyDB(BaseSource):
            name = "my_db"
            async def fetch(self, query: str) -> str:
                return my_db.search(query)

        bot.register_source(MyDB())
        """
        self._sources.register(source)

    # -------------------------------------------------------------------
    # Throttling
    # -------------------------------------------------------------------

    async def _throttle_search(self) -> None:
        now  = time.time()
        wait = (self._last_search_call_ts + self._min_seconds_between_search_calls) - now
        if wait > 0:
            await asyncio.sleep(wait + random.random() * 0.15)
        self._last_search_call_ts = time.time()

    async def _throttle_llm(self) -> None:
        now  = time.time()
        wait = (self._last_llm_call_ts + self._min_seconds_between_llm_calls) - now
        if wait > 0:
            await asyncio.sleep(wait + random.random() * 0.10)
        self._last_llm_call_ts = time.time()

    async def _llm_invoke(self, model_key: str, prompt: str) -> str:
        await self._throttle_llm()
        return await self.get_llm(model_key, "llm").invoke(prompt)

    # -------------------------------------------------------------------
    # Superforecasting preamble (brief but succinct)
    # -------------------------------------------------------------------

    @staticmethod
    def _superforecasting_preamble() -> str:
        return clean_indents(
            """
            ## Superforecasting Protocol

            1. **Outside view first** – anchor to historical base rate for this class of event.
            2. **Inside view** – identify 2-3 causal drivers for and against the outcome.
            3. **Time horizon** – longer horizons regress to base rate; short horizons weight inertia.
            4. **Bias check** – flag availability bias, anchoring, overconfidence.
            5. **Disconfirmation** – what most strongly argues against your lean?
            6. **Synthesise** – blend views; adjust less than feels natural.
            7. **Calibration** – 50% = genuine uncertainty; 5%/95% only with overwhelming evidence.
            """
        ).strip()

    # -------------------------------------------------------------------
    # Research – adaptive, multi-source (Exa + Tavily + AskNews)
    # -------------------------------------------------------------------

    async def _plan_queries(
        self, question: MetaculusQuestion, profile: QuestionProfile
    ) -> list[str]:
        """Build a targeted research query plan for the question (was: _decompose_question)."""
        geo_hint = f" (geography: {profile.geography})" if profile.geography else ""
        prompt = clean_indents(
            f"""
            Build a research plan for a {profile.domain} forecasting question{geo_hint}.

            Return 4 to 6 web-search queries covering: base rates, key drivers,
            recent developments, timelines, expert opinion, prediction market signals.
            Tailor to the {profile.domain} domain{geo_hint}.
            Output ONLY a JSON array of strings.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            """
        )
        try:
            raw = await self._llm_invoke("researcher", prompt)
            raw = raw.strip()
            s, e = raw.find("["), raw.rfind("]")
            if s != -1 and e != -1:
                raw = raw[s : e + 1]
            queries = json.loads(raw)
            if isinstance(queries, list):
                return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:6]
        except Exception:
            pass
        return [
            f"{question.question_text} latest updates{geo_hint}",
            f"{question.question_text} base rate historical frequency",
            f"{question.question_text} prediction market probability",
        ]

    async def _multi_source_research_bundle(
        self, question: MetaculusQuestion, profile: QuestionProfile
    ) -> str:
        llm_queries    = await self._plan_queries(question, profile)
        market_queries = [
            f"metaforecast {question.question_text}",
            f"prediction market odds {question.question_text}",
        ]
        seen: set[str] = set()
        all_queries: list[str] = []
        for q in llm_queries + market_queries:
            q2 = q.strip()
            if q2 and q2 not in seen:
                seen.add(q2); all_queries.append(q2)

        blocks: list[str] = []
        for q in all_queries:
            await self._throttle_search()
            source_results = await self._sources.fetch_all(q)
            blocks.extend(source_results)
        return "\n\n".join(b for b in blocks if b.strip()).strip()

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            cached = await self._research_cache.get(question.page_url)
            if cached:
                return cached

            profile  = await self._analyser.classify(question)
            strategy = ModellingStrategy.select(profile)
            logger.info(
                f"[UpskillBot] '{question.question_text[:60]}…' → "
                f"domain={profile.domain} geo={profile.geography or 'global'} "
                f"strategy={strategy}"
            )

            base = clean_indents(
                f"""
                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria}
                Fine print: {question.fine_print}
                """
            ).strip()

            source_bundle = await self._multi_source_research_bundle(question, profile)
            research_raw  = (
                f"{base}\n\n--- MULTI-SOURCE RESEARCH (Exa / Tavily / AskNews) ---\n{source_bundle}"
                if source_bundle else base
            )

            summarize_prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster working on a {profile.domain} question
                (geography: {profile.geography or 'global'}).
                Summarise the most relevant evidence. Be concise but information-dense.
                Cover: status quo, key drivers, base rates, timelines, market probabilities.

                {research_raw}
                """
            )
            try:
                summary = await self._llm_invoke("summarizer", summarize_prompt)
                final = clean_indents(
                    f"""
                    {base}

                    --- RESEARCH SUMMARY ---
                    {summary}

                    --- RAW RESEARCH ---
                    {source_bundle}
                    """
                ).strip() if source_bundle else f"{base}\n\n--- RESEARCH SUMMARY ---\n{summary}"
            except Exception:
                final = research_raw

            await self._research_cache.set(question.page_url, final)
            return final

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    async def _get_profile_and_strategy(
        self, question: MetaculusQuestion
    ) -> tuple[QuestionProfile, str]:
        profile  = await self._analyser.classify(question)
        strategy = ModellingStrategy.select(profile)
        return profile, strategy

    def _client_context_block(self) -> str:
        if self._client_spec.extra_context:
            return f"\n## Client Context\n{self._client_spec.extra_context}\n"
        return ""

    # -------------------------------------------------------------------
    # Binary
    # -------------------------------------------------------------------

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        profile, strategy = await self._get_profile_and_strategy(question)
        prompt = clean_indents(
            f"""
            You are UpskillBot, a professional superforecaster.
            {self._client_context_block()}
            {self._superforecasting_preamble()}

            {ModellingStrategy.get_prompt_block(strategy, profile)}

            ---

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria (not yet satisfied): {question.resolution_criteria}
            {question.fine_print}
            Research: {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Write exactly 3 paragraphs in first person as UpskillBot, summarizing the key logic that informed this forecast. Do not mention any models, search sources, or research methods.

            {self._get_conditional_disclaimer_if_necessary(question)}
            End with: "Probability: ZZ%" (0-100)
            """
        )
        result = await self._binary_prompt_to_forecast(question, prompt)
        self._validator.validate(question, profile, strategy, result.prediction_value, research)
        return result

    async def _binary_prompt_to_forecast(
        self, question: BinaryQuestion, prompt: str
    ) -> ReasonedPrediction[float]:
        try:
            reasoning = await self._llm_invoke("default", prompt)
        except Exception as exc:
            logger.warning(f"[UpskillBot] LLM failed for {question.page_url}: {exc}. Returning 50% prior.")
            return ReasonedPrediction(prediction_value=0.5, reasoning="LLM failed; returning uninformative prior.")

        logger.info(f"[UpskillBot] Reasoning for {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        raw_p = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        return ReasonedPrediction(
            prediction_value=extremize_probability(raw_p, self._ext_cfg),
            reasoning=reasoning,
        )

    # -------------------------------------------------------------------
    # Multiple choice
    # -------------------------------------------------------------------

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        profile, strategy = await self._get_profile_and_strategy(question)
        prompt = clean_indents(
            f"""
            You are UpskillBot, a professional superforecaster.
            {self._client_context_block()}
            {self._superforecasting_preamble()}

            {ModellingStrategy.get_prompt_block(strategy, profile)}

            ---

            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            {question.fine_print}
            Research: {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Write exactly 3 paragraphs in first person as UpskillBot, summarizing the key logic that informed this forecast. Do not mention any models, search sources, or research methods.

            {self._get_conditional_disclaimer_if_necessary(question)}
            Avoid 0% unless logically impossible.

            End with probabilities in this exact order {question.options}:
            Option_A: Probability_A ...
            """
        )
        result = await self._multiple_choice_prompt_to_forecast(question, prompt)
        self._validator.validate(question, profile, strategy, result.prediction_value, research)
        return result

    async def _multiple_choice_prompt_to_forecast(
        self, question: MultipleChoiceQuestion, prompt: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        reasoning = await self._llm_invoke("default", prompt)
        logger.info(f"[UpskillBot] Reasoning for {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning, output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=f"Option names must match one of: {question.options}. Do not drop any option.",
        )
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    # -------------------------------------------------------------------
    # Numeric
    # -------------------------------------------------------------------

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        profile, strategy = await self._get_profile_and_strategy(question)
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are UpskillBot, a professional superforecaster.
            {self._client_context_block()}
            {self._superforecasting_preamble()}

            {ModellingStrategy.get_prompt_block(strategy, profile)}

            ---

            Question: {question.question_text}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Units: {question.unit_of_measure if question.unit_of_measure else "Not stated (infer)"}
            Research: {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_msg}
            {upper_msg}

            Formatting: no scientific notation; percentiles strictly increasing.

            Write exactly 3 paragraphs in first person as UpskillBot, summarizing the key logic that informed this forecast. Do not mention any models, search sources, or research methods.

            {self._get_conditional_disclaimer_if_necessary(question)}

            End with:
            Percentile 10: XX  Percentile 20: XX  Percentile 40: XX
            Percentile 60: XX  Percentile 80: XX  Percentile 90: XX
            """
        )
        result = await self._numeric_prompt_to_forecast(question, prompt)
        self._validator.validate(question, profile, strategy, result.prediction_value, research)
        return result

    async def _numeric_prompt_to_forecast(
        self, question: NumericQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._llm_invoke("default", prompt)
        logger.info(f"[UpskillBot] Reasoning for {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm"),
            additional_instructions=(
                f'Parse a numeric percentile forecast for: "{question.question_text}"\n'
                f"Units: {question.unit_of_measure}. Convert units if needed."
            ),
            num_validation_samples=self._structure_output_validation_samples,
        )
        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(percentile_list, question),
            reasoning=reasoning,
        )

    # -------------------------------------------------------------------
    # Date
    # -------------------------------------------------------------------

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        profile, strategy = await self._get_profile_and_strategy(question)
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are UpskillBot, a professional superforecaster.
            {self._client_context_block()}
            {self._superforecasting_preamble()}

            {ModellingStrategy.get_prompt_block(strategy, profile)}

            ---

            Question: {question.question_text}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Research: {research}
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            {lower_msg}
            {upper_msg}

            Formatting: dates as YYYY-MM-DD; percentiles chronological and strictly increasing.

            Write exactly 3 paragraphs in first person as UpskillBot, summarizing the key logic that informed this forecast. Do not mention any models, search sources, or research methods.

            {self._get_conditional_disclaimer_if_necessary(question)}

            End with:
            Percentile 10: YYYY-MM-DD  Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD  Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD  Percentile 90: YYYY-MM-DD
            """
        )
        result = await self._date_prompt_to_forecast(question, prompt)
        self._validator.validate(question, profile, strategy, result.prediction_value, research)
        return result

    async def _date_prompt_to_forecast(
        self, question: DateQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._llm_invoke("default", prompt)
        logger.info(f"[UpskillBot] Reasoning for {question.page_url}: {reasoning}")
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning, list[DatePercentile], model=self.get_llm("parser", "llm"),
            additional_instructions=(
                f'Parse a date percentile forecast for: "{question.question_text}"\n'
                "Assume midnight UTC if no time given."
            ),
            num_validation_samples=self._structure_output_validation_samples,
        )
        percentile_list = [
            Percentile(percentile=p.percentile, value=p.value.timestamp())
            for p in date_percentile_list
        ]
        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(percentile_list, question),
            reasoning=reasoning,
        )

    # -------------------------------------------------------------------
    # Bound helpers
    # -------------------------------------------------------------------

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            unit  = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper = question.upper_bound.date().isoformat()
            lower = question.lower_bound.date().isoformat()
            unit  = ""
        else:
            raise ValueError()

        upper_msg = (
            f"The question creator thinks the value is likely not higher than {upper} {unit}."
            if question.open_upper_bound else
            f"The outcome cannot be higher than {upper} {unit}."
        )
        lower_msg = (
            f"The question creator thinks the value is likely not lower than {lower} {unit}."
            if question.open_lower_bound else
            f"The outcome cannot be lower than {lower} {unit}."
        )
        return upper_msg, lower_msg

    # -------------------------------------------------------------------
    # Conditional
    # -------------------------------------------------------------------

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(question.parent,       research,      "parent")
        child_info,  full_research = await self._get_question_prediction_info(question.child,        full_research, "child")
        yes_info,    full_research = await self._get_question_prediction_info(question.question_yes, full_research, "yes")
        no_info,     full_research = await self._get_question_prediction_info(question.question_no,  full_research, "no")

        for info in [parent_info, child_info, yes_info, no_info]:
            pv = getattr(info, "prediction_value", None)
            if isinstance(pv, float):
                info.prediction_value = extremize_probability(pv, self._ext_cfg)  # type: ignore[attr-defined]

        full_reasoning = clean_indents(
            f"""
            ## Parent Reasoning\n{parent_info.reasoning}
            ## Child Reasoning\n{child_info.reasoning}
            ## Yes Reasoning\n{yes_info.reasoning}
            ## No Reasoning\n{no_info.reasoning}
            """
        ).strip()
        return ReasonedPrediction(
            reasoning=full_reasoning,
            prediction_value=ConditionalPrediction(
                parent=parent_info.prediction_value,       # type: ignore
                child=child_info.prediction_value,         # type: ignore
                prediction_yes=yes_info.prediction_value,  # type: ignore
                prediction_no=no_info.prediction_value,    # type: ignore
            ),
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            pf = previous_forecasts[-1]
            if pf.timestamp_end is None or pf.timestamp_end > datetime.now(timezone.utc):
                return (
                    ReasonedPrediction(
                        prediction_value=PredictionAffirmed(),
                        reasoning=f"Reaffirmed at {DataOrganizer.get_readable_prediction(pf)}.",  # type: ignore
                    ),
                    research,
                )  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self, research: str, reasoning: ReasonedPrediction[PredictionTypes], question_type: str
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer
        qt = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {qt} Question Information
            Previously forecasted to: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            Reasoning:
            ```
            {reasoning.reasoning}
            ```
            Do NOT re-forecast the {qt} question.
            """
        ).strip()

    def _get_conditional_disclaimer_if_necessary(self, question: MetaculusQuestion) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return "Forecast ONLY the CHILD question given the parent's resolution. Do not re-forecast the parent."

    # -------------------------------------------------------------------
    # Extremization – top-level sweep (floats skipped; done at source)
    # -------------------------------------------------------------------

    def _extremize_report_if_numeric(self, report: Any) -> None:
        try:
            pv = getattr(report, "prediction_value", None)
            if isinstance(pv, float):
                return   # already extremized at point of creation
            if isinstance(pv, NumericDistribution):
                median_p     = pv.percentile_at_value(pv.median) / 100.0
                extremized_p = extremize_probability(median_p, self._ext_cfg)
                if abs(extremized_p - median_p) > 1e-6:
                    logger.debug(f"[UpskillBot] Numeric extremization: {median_p:.3f} → {extremized_p:.3f}")
        except Exception:
            return

    def _extremize_reports(self, forecast_reports: list[Any]) -> list[Any]:
        for r in forecast_reports:
            self._extremize_report_if_numeric(r)
        return forecast_reports

    async def forecast_on_tournament(self, *args, **kwargs):
        reports = await super().forecast_on_tournament(*args, **kwargs)
        if isinstance(reports, list):
            reports = self._extremize_reports(reports)
            summary = self._validator.summary()
            if summary:
                logger.info(f"[UpskillBot] Validation summary:\n{json.dumps(summary, indent=2)}")
        return reports

    async def forecast_questions(self, *args, **kwargs):
        reports = await super().forecast_questions(*args, **kwargs)
        if isinstance(reports, list):
            reports = self._extremize_reports(reports)
            summary = self._validator.summary()
            if summary:
                logger.info(f"[UpskillBot] Validation summary:\n{json.dumps(summary, indent=2)}")
        return reports


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False

    parser = argparse.ArgumentParser(description="Run UpskillBot – superforecaster bot")
    parser.add_argument(
        "--mode", type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    args    = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    # -----------------------------------------------------------------------
    # Client specialisation – edit this block to tune for a specific client
    # -----------------------------------------------------------------------
    spec = ClientSpecialisation(
        domain_focus=[],           # e.g. ["economics", "finance"]
        trusted_domains=[],        # e.g. ["reuters.com", "ft.com"]
        excluded_domains=[],       # e.g. ["reddit.com"]
        extra_context="",          # inject proprietary intelligence here
        calibration_target=0.15,
    )

    bot = UpskillBot(
        client_spec=spec,
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )

    # -----------------------------------------------------------------------
    # Register additional client sources here, e.g.:
    #   bot.register_source(MyProprietaryDB())
    # -----------------------------------------------------------------------

    client = MetaculusClient()

    if run_mode == "tournament":
        r1 = asyncio.run(bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        r2 = asyncio.run(bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID,       return_exceptions=True))
        r3 = asyncio.run(bot.forecast_on_tournament("market-pulse-26q2",               return_exceptions=True))
        forecast_reports = r1 + r2 + r3

    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)
        )

    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        bot.skip_previously_forecasted_questions = False
        questions        = [client.get_question_by_url(u) for u in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(
            bot.forecast_questions(questions, return_exceptions=True)
        )

    bot.log_report_summary(forecast_reports)
