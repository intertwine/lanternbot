import json

import pytest
import requests

from lanternbot import config
from lanternbot.budget import (
    BudgetCheckFailed,
    BudgetExhausted,
    BudgetGuard,
    CostLedger,
    KeyStatus,
    fetch_openrouter_key_status,
)
from lanternbot.config import BudgetConfig, ModelConfig


def test_project_ids():
    assert config.FALL_2026_TOURNAMENT_ID == 33121
    assert config.MINIBENCH_ID == "minibench"
    assert config.BOT_TESTING_AREA_ID == 32977


def test_budget_config_clamps(monkeypatch):
    monkeypatch.setenv("LANTERN_BUDGET_CAP_USD", "500")
    monkeypatch.setenv("LANTERN_MAX_COST_PER_QUESTION_USD", "9")
    cfg = BudgetConfig.from_env()
    assert cfg.cap_usd == 100.0
    assert cfg.max_cost_per_question_usd == 1.50
    monkeypatch.setenv("LANTERN_BUDGET_CAP_USD", "20")
    assert BudgetConfig.from_env().cap_usd == 20.0


def test_models_must_be_openrouter(monkeypatch):
    ModelConfig.from_env().validate()
    monkeypatch.setenv("LANTERN_FORECASTER_MODELS", "gpt-4o")
    with pytest.raises(ValueError):
        ModelConfig.from_env()
    monkeypatch.setenv("LANTERN_FORECASTER_MODELS", "metaculus/gpt-4o")
    with pytest.raises(ValueError):
        ModelConfig.from_env()
    monkeypatch.setenv("LANTERN_ALLOW_METACULUS_PROXY", "true")
    assert ModelConfig.from_env().forecasters == ("metaculus/gpt-4o",)


def test_default_models_are_priced_in_litellm():
    import litellm

    for m in ModelConfig().all_models():
        assert m in litellm.model_cost, f"{m} has no litellm price -> cost would log as $0"


def test_asknews_credential_names(monkeypatch):
    assert not config.asknews_credentials_present()
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "x")
    assert not config.asknews_credentials_present()  # needs ASKNEWS_SECRET too
    monkeypatch.setenv("ASKNEWS_SECRET", "y")
    assert config.asknews_credentials_present()
    monkeypatch.delenv("ASKNEWS_CLIENT_ID")
    monkeypatch.delenv("ASKNEWS_SECRET")
    monkeypatch.setenv("ASKNEWS_API_KEY", "z")
    assert config.asknews_credentials_present()


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


def test_key_status_parses_and_retries_on_429():
    calls = []
    responses = [
        _Resp(429, headers={"Retry-After": "1"}),
        _Resp(200, {"data": {"usage": 12.5, "limit": 50, "limit_remaining": 37.5, "label": "k"}}),
    ]

    def fake_get(url, headers, timeout):
        calls.append((url, headers))
        return responses.pop(0)

    sleeps = []
    ks = fetch_openrouter_key_status("sk-test", session_get=fake_get, sleep=sleeps.append)
    assert ks.usage_usd == 12.5 and ks.limit_usd == 50 and ks.limit_remaining_usd == 37.5
    assert len(calls) == 2 and sleeps  # retried once
    assert calls[0][0] == "https://openrouter.ai/api/v1/key"


def test_key_status_fails_closed():
    with pytest.raises(BudgetCheckFailed):
        fetch_openrouter_key_status("bad", session_get=lambda *a, **k: _Resp(401), sleep=lambda s: None)

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    with pytest.raises(BudgetCheckFailed):
        fetch_openrouter_key_status("k", session_get=boom, sleep=lambda s: None, attempts=3)


def _guard(tmp_path, usage, limit=None, remaining=None, cap=100.0):
    cfg = BudgetConfig(cap_usd=cap, cost_log_path=str(tmp_path / "c.jsonl"))
    return BudgetGuard(cfg, KeyStatus(usage, limit, remaining), CostLedger(cfg.cost_log_path))


def test_effective_cap_is_min_of_credits_and_100(tmp_path):
    assert _guard(tmp_path, 0, limit=250).effective_cap_usd == 100
    assert _guard(tmp_path, 0, limit=40).effective_cap_usd == 40
    assert _guard(tmp_path, 0, limit=None).effective_cap_usd == 100


def test_guard_stops_before_cap(tmp_path):
    g = _guard(tmp_path, usage=98.0)
    g.check_can_start_question()  # 2.00 left >= 1.50
    assert g.per_question_hard_limit() == 1.5
    g.record_question(question_url="u", question_id=1, question_type="BinaryQuestion",
                      project="p", litellm_cost_usd=0.6, research_calls=0,
                      status="ok", submitted=False)
    with pytest.raises(BudgetExhausted):
        g.check_can_start_question()  # 1.40 left < 1.50
    assert g.stopped
    rec = json.loads(open(tmp_path / "c.jsonl").read().splitlines()[0])
    assert rec["total_cost_usd"] == 0.6 and rec["cycle_spent_usd"] == 98.6


def test_key_remaining_is_respected(tmp_path):
    g = _guard(tmp_path, usage=10, limit=100, remaining=1.0)
    with pytest.raises(BudgetExhausted):
        g.check_can_start_question()


def test_unpriced_search_fee_added(tmp_path):
    g = _guard(tmp_path, usage=0)
    total = g.record_question(question_url="u", question_id=1, question_type="X", project="p",
                              litellm_cost_usd=0.10, research_calls=2, status="ok", submitted=True)
    assert abs(total - 0.13) < 1e-9
