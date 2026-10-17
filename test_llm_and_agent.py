import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

import agent
import evaluate
from agent import calculate, run_tool
from llm import cost_in_dollars, is_retryable, with_retry


def api_error(status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    error_class = {429: anthropic.RateLimitError, 400: anthropic.BadRequestError}.get(
        status, anthropic.APIStatusError
    )
    return error_class("test error", response=response, body=None)


# --- Retry with backoff ----------------------------------------------------


@pytest.mark.parametrize("status, expected", [(429, True), (500, True), (529, True), (400, False), (401, False)])
def test_is_retryable(status: int, expected: bool) -> None:
    assert is_retryable(api_error(status)) is expected


def test_retry_recovers_after_temporary_failures() -> None:
    attempts = []
    waits: list[float] = []

    @with_retry(max_attempts=4, base_delay=1.0, sleep=waits.append)
    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise api_error(529)
        return "ok"

    assert flaky() == "ok"
    assert len(attempts) == 3
    # Exponential backoff with jitter: ~1s then ~2s (each scaled by 0.5-1.0).
    assert 0.5 <= waits[0] <= 1.0 and 1.0 <= waits[1] <= 2.0


def test_retry_gives_up_after_max_attempts() -> None:
    @with_retry(max_attempts=3, sleep=lambda _: None)
    def always_overloaded() -> None:
        raise api_error(529)

    with pytest.raises(anthropic.APIStatusError):
        always_overloaded()


def test_retry_does_not_retry_bad_requests() -> None:
    attempts = []

    @with_retry(max_attempts=4, sleep=lambda _: None)
    def bad_request() -> None:
        attempts.append(1)
        raise api_error(400)

    with pytest.raises(anthropic.BadRequestError):
        bad_request()
    assert len(attempts) == 1


def test_cost_in_dollars() -> None:
    cost = cost_in_dollars("claude-opus-5-5", input_tokens=1_000_000, output_tokens=100_000)
    assert cost == pytest.approx({"input": 4.0, "output": 2.0})
    assert cost_in_dollars("some-unknown-model", 10, 10) is None


# --- Calculator tool -------------------------------------------------------


@pytest.mark.parametrize(
    "expression, expected",
    [("3 * 35", 105), ("(220 * 12) * 0.9", 2376), ("-4 + 10 / 4", -1.5), ("2 ** 10", 1024), ("17 % 5", 2)],
)
def test_calculator_does_arithmetic(expression: str, expected: float) -> None:
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "open('secrets.txt')", "x + 1", "[1, 2]", "9 ** 999999"],
)
def test_calculator_refuses_anything_but_arithmetic(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)


def test_tool_errors_are_returned_to_the_model_not_raised() -> None:
    text, is_error = run_tool("calculator", {"expression": "1 / 0"})
    assert is_error and text.startswith("Error:")


# --- LangGraph agent loop (with a fake Claude, so no API key is needed) ----


def fake_response(stop_reason: str, *blocks: Any) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


def test_agent_uses_a_tool_then_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [
            fake_response(
                "tool_use",
                SimpleNamespace(type="tool_use", id="call_1", name="calculator", input={"expression": "3 * 35"}),
            ),
            fake_response("end_turn", SimpleNamespace(type="text", text="That would be £105.")),
        ]
    )
    seen_messages: list[list[dict]] = []

    def fake_call_claude(messages: list[dict], **kwargs: Any) -> SimpleNamespace:
        seen_messages.append(list(messages))
        return next(replies)

    monkeypatch.setattr(agent, "call_claude", fake_call_claude)
    answer, tools_used = agent.run_agent("What do 3 extra hours in Cedar cost?")

    assert answer == "That would be £105."
    assert tools_used == ["calculator"]
    # On the second call, Claude was sent the calculator's result.
    tool_result = seen_messages[1][-1]["content"][0]
    assert tool_result["tool_use_id"] == "call_1" and tool_result["content"] == "105"


def test_agent_stops_after_the_model_call_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    looping = fake_response(
        "tool_use", SimpleNamespace(type="tool_use", id="c", name="calculator", input={"expression": "1+1"})
    )
    calls = []
    monkeypatch.setattr(agent, "call_claude", lambda **kwargs: calls.append(1) or looping)
    answer, _ = agent.run_agent("loop forever")
    assert len(calls) == agent.MAX_MODEL_CALLS
    assert "couldn't finish" in answer


# --- Judge -----------------------------------------------------------------


def test_judge_parses_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = json.dumps({"reason": "Gives the right price.", "score": 5})
    monkeypatch.setattr(
        evaluate,
        "call_claude",
        lambda **kwargs: fake_response("end_turn", SimpleNamespace(type="text", text=reply)),
    )
    judgement = evaluate.judge("How much is Resident?", "£220 per month.", "It's £220 a month.")
    assert judgement == evaluate.Judgement(score=5, reason="Gives the right price.")


def test_judge_sees_the_whole_handbook() -> None:
    # Without the handbook, the judge marked true facts missing from the notes as invented.
    prompt = evaluate.judge_system_prompt()
    assert "<handbook>" in prompt
    assert "10% discount" in prompt and "Friday at 4pm" in prompt


def test_eval_set_has_at_least_eight_questions() -> None:
    cases = evaluate.load_eval_set()
    assert len(cases) >= 8
    assert all(case.question and case.good_answer for case in cases)
