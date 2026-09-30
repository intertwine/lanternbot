"""
Budget guard and per-question cost ledger.

How the cap is enforced
-----------------------
GitHub Actions runners are stateless, so a local running total would reset on
every 20-minute run. The authoritative cumulative spend is therefore read from
OpenRouter itself (GET https://openrouter.ai/api/v1/key -> data.usage, the
key's all-time spend, and data.limit, the key's credit limit) at the start of
every run. Within a run, the litellm cost of each finished question
(forecasting-tools' MonetaryCostManager) is added on top.

    effective_cap = min(LANTERN_BUDGET_CAP_USD (<= $100), key limit if set)
    before each question: stop if spent + max_cost_per_question > effective_cap
    during each question: MonetaryCostManager hard limit = min($1.50, remaining)

If OpenRouter cannot be reached the guard FAILS CLOSED (no forecasting).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from lanternbot.config import BudgetConfig

logger = logging.getLogger(__name__)

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"

# Perplexity-style search models charge a per-request fee that litellm's cost
# table does not include. We add this estimate per research call to the
# logged cost so the ledger is closer to list price (OpenRouter usage stays
# the authoritative number for the cap).
DEFAULT_UNPRICED_FEE_PER_RESEARCH_CALL_USD = 0.015


class BudgetExhausted(Exception):
    """Raised when the cycle cap would be exceeded."""


class BudgetCheckFailed(Exception):
    """Raised when the authoritative spend cannot be determined (fail closed)."""


@dataclass
class KeyStatus:
    usage_usd: float
    limit_usd: float | None
    limit_remaining_usd: float | None
    label: str | None = None


def fetch_openrouter_key_status(
    api_key: str,
    *,
    session_get: Callable[..., Any] = requests.get,
    attempts: int = 4,
    base_sleep_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> KeyStatus:
    """Read-only call to OpenRouter's key endpoint, with retry on 429/5xx."""
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = session_get(
                OPENROUTER_KEY_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=20,
            )
            status = getattr(resp, "status_code", 200)
            if status == 429 or status >= 500:
                raise requests.HTTPError(f"HTTP {status}", response=resp)
            if status >= 400:
                # 401/403: bad key. Don't retry.
                raise BudgetCheckFailed(f"OpenRouter key check returned HTTP {status}")
            data = resp.json().get("data", {})
            return KeyStatus(
                usage_usd=float(data.get("usage") or 0.0),
                limit_usd=(None if data.get("limit") is None else float(data["limit"])),
                limit_remaining_usd=(
                    None
                    if data.get("limit_remaining") is None
                    else float(data["limit_remaining"])
                ),
                label=data.get("label"),
            )
        except BudgetCheckFailed:
            raise
        except Exception as e:  # network / 429 / 5xx / bad JSON
            last_err = e
            wait = base_sleep_s * (2**attempt)
            retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
            try:
                wait = max(wait, float(retry_after.get("Retry-After", 0)))
            except (TypeError, ValueError):
                pass
            if attempt == attempts - 1:
                break
            logger.warning(f"OpenRouter key check failed ({e}); retrying in {wait:.0f}s")
            sleep(wait)
    raise BudgetCheckFailed(f"Could not read OpenRouter key status: {last_err}")


class CostLedger:
    """Append-only JSONL log, one line per question attempt."""

    def __init__(self, path: str) -> None:
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        record = {"ts_utc": datetime.now(timezone.utc).isoformat(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


class BudgetGuard:
    def __init__(
        self,
        config: BudgetConfig,
        key_status: KeyStatus,
        ledger: CostLedger | None = None,
        unpriced_fee_per_research_call_usd: float = DEFAULT_UNPRICED_FEE_PER_RESEARCH_CALL_USD,
    ) -> None:
        self.config = config
        self.start_usage_usd = key_status.usage_usd
        caps = [config.cap_usd]
        if key_status.limit_usd is not None:
            caps.append(key_status.limit_usd)
        self.effective_cap_usd = min(caps)
        # If the key's own remaining credit is lower than what our arithmetic
        # says, trust the key.
        self._key_remaining = key_status.limit_remaining_usd
        self.run_spend_usd = 0.0
        self.questions_attempted = 0
        self.ledger = ledger or CostLedger(config.cost_log_path)
        self.unpriced_fee_per_research_call_usd = unpriced_fee_per_research_call_usd
        self.stopped = False

    @property
    def spent_usd(self) -> float:
        return self.start_usage_usd + self.run_spend_usd

    @property
    def remaining_usd(self) -> float:
        rem = self.effective_cap_usd - self.spent_usd
        if self._key_remaining is not None:
            rem = min(rem, self._key_remaining - self.run_spend_usd)
        return max(0.0, rem)

    def per_question_hard_limit(self) -> float:
        return min(self.config.max_cost_per_question_usd, self.remaining_usd)

    def check_can_start_question(self) -> None:
        """Only start a question if a worst-case question still fits under the cap."""
        if self.remaining_usd < self.config.max_cost_per_question_usd:
            self.stopped = True
            raise BudgetExhausted(
                f"Budget guard: spent ${self.spent_usd:.4f} of cap "
                f"${self.effective_cap_usd:.2f}; remaining ${self.remaining_usd:.4f} "
                f"< per-question max ${self.config.max_cost_per_question_usd:.2f}. "
                "Stopping forecasting."
            )

    def record_question(
        self,
        *,
        question_url: str | None,
        question_id: int | None,
        question_type: str,
        project: str | None,
        litellm_cost_usd: float,
        research_calls: int,
        status: str,
        submitted: bool,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> float:
        est_fees = research_calls * self.unpriced_fee_per_research_call_usd
        total = float(litellm_cost_usd) + est_fees
        self.run_spend_usd += total
        self.questions_attempted += 1
        self.ledger.write(
            {
                "question_url": question_url,
                "question_id": question_id,
                "question_type": question_type,
                "project": project,
                "status": status,
                "submitted": submitted,
                "litellm_cost_usd": round(float(litellm_cost_usd), 6),
                "est_unpriced_fees_usd": round(est_fees, 6),
                "total_cost_usd": round(total, 6),
                "over_target": total > self.config.max_cost_per_question_usd,
                "cycle_spent_usd": round(self.spent_usd, 6),
                "cycle_cap_usd": self.effective_cap_usd,
                "error": error,
                **(extra or {}),
            }
        )
        return total

    def summary(self) -> dict[str, Any]:
        return {
            "start_usage_usd": round(self.start_usage_usd, 6),
            "run_spend_usd": round(self.run_spend_usd, 6),
            "spent_usd": round(self.spent_usd, 6),
            "effective_cap_usd": self.effective_cap_usd,
            "remaining_usd": round(self.remaining_usd, 6),
            "questions_attempted": self.questions_attempted,
            "stopped_by_budget": self.stopped,
        }
