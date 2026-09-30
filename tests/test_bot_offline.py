"""
Offline end-to-end dry runs of LanternBot with mocked LLMs and a mocked
Metaculus HTTP layer. Nothing touches the network (see conftest.py).
"""
import asyncio
import json
from datetime import datetime, timezone

import pytest
from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    DatePercentile,
    DateQuestion,
    GeneralLlm,
    MonetaryCostManager,
    MultipleChoiceQuestion,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.questions import DiscreteQuestion
import forecasting_tools.helpers.metaculus_client as mc_mod

import main as bot_main
from lanternbot.budget import BudgetGuard, CostLedger, KeyStatus
from lanternbot.config import BudgetConfig, ModelConfig


# ------------------------------------------------------------------ fakes ---
class FakeHttp:
    def __init__(self):
        self.posts = []

    def post(self, url, json=None, **kwargs):
        self.posts.append((url, json))

        class R:
            status_code = 201
            ok = True
            text = "{}"
            content = b"{}"
            reason = "Created"

            def raise_for_status(self):
                return None

            def json(self):
                return {}

        return R()

    def get(self, *a, **k):
        raise AssertionError("No GETs to Metaculus expected in offline tests")


@pytest.fixture
def fake_http(monkeypatch):
    http = FakeHttp()
    monkeypatch.setattr(mc_mod.requests, "post", http.post)
    monkeypatch.setattr(mc_mod.requests, "get", http.get)
    monkeypatch.setattr(mc_mod.MetaculusClient, "_sleep_between_requests", lambda self: None)
    monkeypatch.setenv("METACULUS_TOKEN", "offline-test-token")
    return http


@pytest.fixture
def fake_llm(monkeypatch):
    state = {"cost_per_call": 0.01, "models": []}

    async def invoke(self, prompt, *a, **k):
        MonetaryCostManager.raise_error_if_limit_would_be_reached()  # mimics litellm pre-call hook
        state["models"].append(self.model)
        MonetaryCostManager.increase_current_usage_in_parent_managers(state["cost_per_call"])
        return "Some reasoning. Probability: 30%"

    monkeypatch.setattr(GeneralLlm, "invoke", invoke)

    async def fake_structure_output(text=None, output_type=None, *a, text_to_structure=None, **k):
        q = state["question"]
        if output_type is BinaryPrediction:
            return BinaryPrediction(prediction_in_decimal=0.3)
        if output_type is PredictedOptionList:
            n = len(q.options)
            return PredictedOptionList(
                predicted_options=[PredictedOption(option_name=o, probability=1 / n) for o in q.options]
            )
        if output_type == list[Percentile]:
            lo, hi = q.lower_bound, q.upper_bound
            return [
                Percentile(percentile=p, value=lo + (hi - lo) * f)
                for p, f in [(0.1, 0.15), (0.2, 0.25), (0.4, 0.4), (0.6, 0.55), (0.8, 0.7), (0.9, 0.8)]
            ]
        if output_type == list[DatePercentile]:
            lo, hi = q.lower_bound.timestamp(), q.upper_bound.timestamp()
            return [
                DatePercentile(percentile=p, value=datetime.fromtimestamp(lo + (hi - lo) * f, tz=timezone.utc))
                for p, f in [(0.1, 0.15), (0.2, 0.25), (0.4, 0.4), (0.6, 0.55), (0.8, 0.7), (0.9, 0.8)]
            ]
        raise AssertionError(f"unexpected output type {output_type}")

    monkeypatch.setattr(bot_main, "structure_output", fake_structure_output)
    return state


def _questions():
    common = dict(resolution_criteria="rc", fine_print="fp", background_info="bg")
    return [
        BinaryQuestion(question_text="Binary?", id_of_post=101, id_of_question=1001,
                       page_url="https://www.metaculus.com/questions/101/", already_forecasted=False, **common),
        MultipleChoiceQuestion(question_text="MC?", id_of_post=102, id_of_question=1002, options=["A", "B", "C"],
                               page_url="https://www.metaculus.com/questions/102/", already_forecasted=False, **common),
        NumericQuestion(question_text="Numeric?", id_of_post=103, id_of_question=1003, lower_bound=0, upper_bound=100,
                        open_lower_bound=False, open_upper_bound=True, unit_of_measure="units",
                        page_url="https://www.metaculus.com/questions/103/", already_forecasted=False, **common),
        DiscreteQuestion(question_text="Discrete?", id_of_post=104, id_of_question=1004, lower_bound=-0.5,
                         upper_bound=10.5, open_lower_bound=False, open_upper_bound=False, cdf_size=12,
                         unit_of_measure="count",
                         page_url="https://www.metaculus.com/questions/104/", already_forecasted=False, **common),
        DateQuestion(question_text="Date?", id_of_post=105, id_of_question=1005,
                     lower_bound=datetime(2026, 10, 1, tzinfo=timezone.utc),
                     upper_bound=datetime(2027, 6, 1, tzinfo=timezone.utc),
                     open_lower_bound=False, open_upper_bound=True,
                     page_url="https://www.metaculus.com/questions/105/", already_forecasted=False, **common),
        BinaryQuestion(question_text="Already done", id_of_post=106, id_of_question=1006,
                       page_url="https://www.metaculus.com/questions/106/", already_forecasted=True, **common),
    ]


def _make_bot(tmp_path, usage=0.0, limit=None, publish=True):
    cfg = BudgetConfig(cap_usd=100.0, cost_log_path=str(tmp_path / "cost_log.jsonl"))
    guard = BudgetGuard(cfg, KeyStatus(usage, limit, None), CostLedger(cfg.cost_log_path))
    llms, forecasters, paid = bot_main.build_llms(ModelConfig())
    bot = bot_main.LanternBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        publish_reports_to_metaculus=publish,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=llms,
        metaculus_client=bot_main.PrivateCommentMetaculusClient(),
        budget_guard=guard,
        forecaster_llms=forecasters,
        research_is_paid_search=paid,
    )
    return bot, guard


def _run(bot, questions, state):
    # structure_output fake needs to know the current question; bot runs
    # questions sequentially, so set it per question.
    reports = []
    for q in questions:
        state["question"] = q
        reports += asyncio.run(bot.forecast_questions([q], return_exceptions=True))
    return reports


def _ledger(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "cost_log.jsonl").read().splitlines()]


# ------------------------------------------------------------------ tests ---
def test_all_question_types_forecast_and_private_comment(tmp_path, fake_http, fake_llm):
    bot, guard = _make_bot(tmp_path)
    qs = _questions()
    reports = _run(bot, qs, fake_llm)
    errors = [r for r in reports if isinstance(r, BaseException)]
    assert not errors, errors
    assert len(reports) == 5  # already-forecasted question skipped

    comment_posts = [j for u, j in fake_http.posts if u.endswith("/comments/create/")]
    forecast_posts = [(u, j) for u, j in fake_http.posts if "/comments/create/" not in u]
    assert len(comment_posts) == 5 and len(forecast_posts) == 5
    assert all(c["is_private"] is True for c in comment_posts)
    assert {c["on_post"] for c in comment_posts} == {101, 102, 103, 104, 105}
    assert all(c["text"] for c in comment_posts)

    ledger = _ledger(tmp_path)
    assert [r["question_type"] for r in ledger] == [
        "BinaryQuestion", "MultipleChoiceQuestion", "NumericQuestion", "DiscreteQuestion", "DateQuestion"]
    assert all(r["status"] == "ok" and r["submitted"] for r in ledger)
    assert all(0 < r["total_cost_usd"] <= 1.5 for r in ledger)

    # forecasts rotate across all configured OpenRouter models
    used = set(fake_llm["models"])
    assert set(ModelConfig().forecasters) <= used
    assert all(m.startswith("openrouter/") for m in used)


def test_private_comment_client_forces_private(fake_http):
    c = bot_main.PrivateCommentMetaculusClient()
    c.post_question_comment(1, "x", is_private=False)
    assert fake_http.posts[-1][1]["is_private"] is True


def test_budget_guard_stops_mid_run(tmp_path, fake_http, fake_llm):
    fake_llm["cost_per_call"] = 0.10  # ~ $0.70-0.80 per question
    bot, guard = _make_bot(tmp_path, usage=97.0)
    reports = _run(bot, _questions()[:5], fake_llm)
    ok = [r for r in reports if not isinstance(r, BaseException)]
    # 7 LLM calls x $0.10 + $0.015 est. search fee = $0.715/question.
    # $3.00 left: q1 -> 2.285, q2 -> 1.57 (still >= 1.50), q3 -> 0.855 < 1.50 -> stop
    assert len(ok) == 3
    assert len(_ledger(tmp_path)) == 3  # 4th and 5th never started
    assert guard.stopped
    assert guard.spent_usd <= 100.0
    assert len([u for u, _ in fake_http.posts if "comments" in u]) == len(ok)


def test_per_question_hard_limit_aborts_expensive_question(tmp_path, fake_http, fake_llm):
    fake_llm["cost_per_call"] = 0.60  # would be > $1.50 per question
    bot, guard = _make_bot(tmp_path)
    reports = _run(bot, _questions()[:1], fake_llm)
    assert isinstance(reports[0], BaseException)
    assert fake_http.posts == []  # nothing submitted
    rec = _ledger(tmp_path)[0]
    assert rec["status"] == "error" and rec["submitted"] is False
    assert rec["litellm_cost_usd"] <= 1.5 + 0.6  # stopped at the limit (one call of overshoot max)


def test_no_publish_mode_posts_nothing(tmp_path, fake_http, fake_llm):
    bot, _ = _make_bot(tmp_path, publish=False)
    reports = _run(bot, _questions()[:2], fake_llm)
    assert all(not isinstance(r, BaseException) for r in reports)
    assert fake_http.posts == []


def test_main_exits_cleanly_without_openrouter_key(monkeypatch, capsys):
    monkeypatch.setenv("METACULUS_TOKEN", "offline-test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-personal")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-personal")

    def explode(*a, **k):
        raise AssertionError("must not touch Metaculus or build a bot")

    monkeypatch.setattr(bot_main, "LanternBot", explode)
    monkeypatch.setattr(bot_main, "PrivateCommentMetaculusClient", explode)
    assert bot_main.main(["--mode", "tournament"]) == 0
    out = capsys.readouterr().out
    assert "OPENROUTER_API_KEY is not set" in out
    import os
    assert "OPENAI_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ


def test_main_fails_closed_when_budget_unknown(monkeypatch, capsys):
    monkeypatch.setenv("METACULUS_TOKEN", "offline-test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")

    def explode(*a, **k):
        raise AssertionError("must not build a bot when budget is unknown")

    monkeypatch.setattr(bot_main, "LanternBot", explode)
    import lanternbot.budget as b
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    # network is blocked by conftest -> OpenRouter check fails -> exit 1, no forecasting
    assert bot_main.main(["--mode", "test_questions"]) == 1
    assert "fail closed" in capsys.readouterr().out


def test_test_mode_targets_bot_testing_area_only(monkeypatch, tmp_path):
    monkeypatch.setenv("METACULUS_TOKEN", "offline-test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setenv("LANTERN_COST_LOG_PATH", str(tmp_path / "cost_log.jsonl"))
    import lanternbot.budget as b
    monkeypatch.setattr(b, "fetch_openrouter_key_status", lambda k: KeyStatus(0.0, 50.0, 50.0))
    targets = []

    async def fake_fot(self, tid, return_exceptions=False):
        targets.append(tid)
        return []

    monkeypatch.setattr(bot_main.LanternBot, "forecast_on_tournament", fake_fot)
    assert bot_main.main(["--mode", "test_questions"]) == 0
    assert targets == [32977]
    targets.clear()
    assert bot_main.main(["--mode", "tournament"]) == 0
    assert targets == [33121, "minibench"]
    summary = json.load(open(tmp_path / "run_summary.json"))
    assert summary["effective_cap_usd"] == 50.0
