import pytest

from reposcope.llm import LLMResponse, ToolCall, UserMessage
from tests.fakes import FakeLLM


def test_fake_llm_returns_scripted_responses_in_order() -> None:
    first = LLMResponse(text="", tool_calls=(ToolCall("grep_code", {"pattern": "Job"}),))
    second = LLMResponse(text="done")
    llm = FakeLLM([first, second])

    assert llm.generate("system", [UserMessage("question")], []) is first
    assert llm.generate("system", [UserMessage("question")], []) is second


def test_fake_llm_records_the_messages_it_was_given() -> None:
    llm = FakeLLM([LLMResponse(text="done")])
    question = UserMessage("how is a job submitted?")

    llm.generate("system", [question], [])

    assert llm.requests == [[question]]


def test_fake_llm_fails_loudly_when_the_script_runs_out() -> None:
    llm = FakeLLM([])

    with pytest.raises(AssertionError, match="more often than it has scripted responses"):
        llm.generate("system", [UserMessage("question")], [])
