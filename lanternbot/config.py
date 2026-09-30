"""
Central configuration for lanternbot.

Everything that affects money, targets, or which models are called lives here,
so it can be reviewed in one place. Values can be overridden with environment
variables (GitHub Actions repository *variables*, not secrets, are fine for the
non-sensitive ones).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Metaculus project IDs (Fall 2026)
# --------------------------------------------------------------------------- #
# Fall 2026 FutureEval bot tournament.
#   Verified 2026-09-30: the Fall 2026 announcement page embeds
#   {"id": 33121, "slug": "fall-futureeval-2026"}, and forecasting-tools 0.3.1
#   ships MetaculusClient.FE_FALL_2026_ID = 33121 (= CURRENT_AI_COMPETITION_ID).
FALL_2026_TOURNAMENT_ID: int = 33121
FALL_2026_TOURNAMENT_SLUG: str = "fall-futureeval-2026"

# MiniBench. forecasting-tools 0.3.1: CURRENT_MINIBENCH_ID = "minibench".
#   The slug is what Metaculus's own template uses; the numeric ID of the
#   currently active MiniBench round could NOT be verified without an API token
#   (UNVERIFIED numeric ID; slug is the documented way to target it).
MINIBENCH_ID: str = "minibench"

# Bot testing area. Verified 2026-09-30 in the Fall 2026 announcement:
#   'Use the ID "bot-testing-area" or "32977"'.
BOT_TESTING_AREA_ID: int = 32977
BOT_TESTING_AREA_SLUG: str = "bot-testing-area"

TOURNAMENT_URLS = {
    "tournament": f"https://www.metaculus.com/tournament/{FALL_2026_TOURNAMENT_SLUG}/",
    "test_questions": f"https://www.metaculus.com/tournament/{BOT_TESTING_AREA_SLUG}/",
}

# --------------------------------------------------------------------------- #
# Money
# --------------------------------------------------------------------------- #
# Hard cap for the whole E03 cycle (testing included). The effective cap is
# min(this, the OpenRouter key's own credit limit), i.e. min(sponsored credits,
# $100). Override with LANTERN_BUDGET_CAP_USD (can only be *lowered* in code
# review; values above 100 are clamped, see BudgetConfig.from_env).
DEFAULT_BUDGET_CAP_USD: float = 100.0
ABSOLUTE_MAX_BUDGET_CAP_USD: float = 100.0

# Target / hard ceiling per question. A question whose LLM spend would exceed
# this is aborted (MonetaryCostManager hard limit) and not submitted.
DEFAULT_MAX_COST_PER_QUESTION_USD: float = 1.50

# Where the per-question cost ledger is written (uploaded as an Actions artifact).
DEFAULT_COST_LOG_PATH: str = "cost_logs/cost_log.jsonl"

# --------------------------------------------------------------------------- #
# Models (all through OpenRouter; nothing else is allowed)
# --------------------------------------------------------------------------- #
# Forecasts rotate round-robin through this list (one OpenAI, one Anthropic,
# one Google model). Names must exist on OpenRouter; they are priced in
# litellm's cost table (checked against litellm 1.103.1), which is needed for
# per-question cost tracking. UNVERIFIED: whether Metaculus's sponsored key
# allows every one of these models.
DEFAULT_FORECASTER_MODELS: tuple[str, ...] = (
    "openrouter/openai/gpt-5.4-mini",
    "openrouter/anthropic/claude-sonnet-4.6",
    "openrouter/google/gemini-3.5-flash",
)
DEFAULT_PARSER_MODEL: str = "openrouter/openai/gpt-4.1-mini"
DEFAULT_SUMMARIZER_MODEL: str = "openrouter/openai/gpt-4.1-mini"
# Used for research only when AskNews credentials are absent.
DEFAULT_RESEARCH_MODEL: str = "openrouter/perplexity/sonar"

PREDICTIONS_PER_RESEARCH_REPORT: int = 5
RESEARCH_REPORTS_PER_QUESTION: int = 1
LLM_TIMEOUT_SECONDS: int = 120
LLM_ALLOWED_TRIES: int = 3  # retries with backoff inside forecasting-tools

# Environment variables that could route spend to a personal account. They are
# removed from the process environment at startup so forecasting-tools'
# default-model logic can never fall back to them.
PERSONAL_KEY_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PERPLEXITY_API_KEY",
    "EXA_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)

_PLACEHOLDERS = {"", "REPLACE_ME", "1234567890", "your-token-here", "your-api-key-here"}


def env_is_set(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip() and val.strip() not in _PLACEHOLDERS)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BudgetConfig:
    cap_usd: float = DEFAULT_BUDGET_CAP_USD
    max_cost_per_question_usd: float = DEFAULT_MAX_COST_PER_QUESTION_USD
    cost_log_path: str = DEFAULT_COST_LOG_PATH

    @classmethod
    def from_env(cls) -> "BudgetConfig":
        cap = _float_env("LANTERN_BUDGET_CAP_USD", DEFAULT_BUDGET_CAP_USD)
        cap = max(0.0, min(cap, ABSOLUTE_MAX_BUDGET_CAP_USD))
        per_q = _float_env(
            "LANTERN_MAX_COST_PER_QUESTION_USD", DEFAULT_MAX_COST_PER_QUESTION_USD
        )
        per_q = max(0.0, min(per_q, DEFAULT_MAX_COST_PER_QUESTION_USD))
        path = os.getenv("LANTERN_COST_LOG_PATH") or DEFAULT_COST_LOG_PATH
        return cls(cap_usd=cap, max_cost_per_question_usd=per_q, cost_log_path=path)


@dataclass(frozen=True)
class ModelConfig:
    forecasters: tuple[str, ...] = field(default=DEFAULT_FORECASTER_MODELS)
    parser: str = DEFAULT_PARSER_MODEL
    summarizer: str = DEFAULT_SUMMARIZER_MODEL
    research_model: str = DEFAULT_RESEARCH_MODEL
    allow_metaculus_proxy: bool = False

    @classmethod
    def from_env(cls) -> "ModelConfig":
        raw = os.getenv("LANTERN_FORECASTER_MODELS", "")
        forecasters = tuple(m.strip() for m in raw.split(",") if m.strip()) or DEFAULT_FORECASTER_MODELS
        cfg = cls(
            forecasters=forecasters,
            parser=os.getenv("LANTERN_PARSER_MODEL") or DEFAULT_PARSER_MODEL,
            summarizer=os.getenv("LANTERN_SUMMARIZER_MODEL") or DEFAULT_SUMMARIZER_MODEL,
            research_model=os.getenv("LANTERN_RESEARCH_MODEL") or DEFAULT_RESEARCH_MODEL,
            allow_metaculus_proxy=_truthy_env("LANTERN_ALLOW_METACULUS_PROXY"),
        )
        cfg.validate()
        return cfg

    def all_models(self) -> list[str]:
        return [*self.forecasters, self.parser, self.summarizer, self.research_model]

    def validate(self) -> None:
        for m in self.all_models():
            if m.startswith("openrouter/"):
                continue
            if m.startswith("metaculus/") and self.allow_metaculus_proxy:
                continue
            raise ValueError(
                f"Model {m!r} is not an OpenRouter model. lanternbot only spends "
                "through the Metaculus-sponsored OpenRouter key (no personal-key "
                "fallback). The Metaculus proxy ('metaculus/...') is only allowed "
                "when LANTERN_ALLOW_METACULUS_PROXY=true."
            )


def asknews_credentials_present() -> bool:
    """forecasting-tools' AskNewsSearcher accepts EITHER the OAuth pair
    ASKNEWS_CLIENT_ID + ASKNEWS_SECRET (preferred if both kinds are set) OR a
    single ASKNEWS_API_KEY. Note the name is ASKNEWS_SECRET, not
    ASKNEWS_CLIENT_SECRET."""
    return (env_is_set("ASKNEWS_CLIENT_ID") and env_is_set("ASKNEWS_SECRET")) or env_is_set(
        "ASKNEWS_API_KEY"
    )
