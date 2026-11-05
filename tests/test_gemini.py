import pytest
from google.genai import types

from reposcope.gemini import (
    GeminiLLM,
    normalise_arguments,
    to_contents,
    to_response,
    to_tool,
)
from reposcope.llm import (
    LLMError,
    LLMResponse,
    ToolResult,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)


def response_with(
    *parts: types.Part, finish_reason: str | None = None
) -> types.GenerateContentResponse:
    content = types.Content(role="model", parts=list(parts))
    candidate = types.Candidate(content=content, finish_reason=finish_reason)
    return types.GenerateContentResponse(candidates=[candidate])


def call_part(name: str, args: dict[str, object]) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def test_normalise_turns_whole_number_floats_into_ints() -> None:
    result = normalise_arguments({"start_line": 10.0, "end_line": 20.0})

    assert result == {"start_line": 10, "end_line": 20}
    assert all(type(value) is int for value in result.values())


def test_normalise_leaves_other_values_alone() -> None:
    arguments = {"pattern": "Job", "ratio": 0.5, "verbose": True, "count": 3}

    assert normalise_arguments(arguments) == arguments


def test_response_text_is_collected_from_text_parts() -> None:
    response = to_response(response_with(types.Part(text="It starts in "), types.Part(text="Job.")))

    assert response.text == "It starts in Job."
    assert response.tool_calls == ()


def test_response_ignores_thought_parts() -> None:
    response = to_response(
        response_with(
            types.Part(text="private reasoning", thought=True), types.Part(text="Answer.")
        )
    )

    assert response.text == "Answer."


def test_response_exposes_tool_calls_with_integer_arguments() -> None:
    response = to_response(
        response_with(
            call_part("read_file", {"path": "Job.java", "start_line": 10.0, "end_line": 20.0})
        )
    )

    assert [(call.name, call.arguments) for call in response.tool_calls] == [
        ("read_file", {"path": "Job.java", "start_line": 10, "end_line": 20})
    ]


def test_response_supports_several_tool_calls_in_one_turn() -> None:
    response = to_response(
        response_with(
            call_part("grep_code", {"pattern": "a"}), call_part("grep_code", {"pattern": "b"})
        )
    )

    assert [call.arguments["pattern"] for call in response.tool_calls] == ["a", "b"]


def test_response_keeps_the_original_content_for_replay() -> None:
    gemini_response = response_with(call_part("grep_code", {"pattern": "Job"}))

    response = to_response(gemini_response)

    assert response.raw is gemini_response.candidates[0].content


def test_response_without_candidates_is_an_error() -> None:
    with pytest.raises(LLMError, match="no candidates"):
        to_response(types.GenerateContentResponse(candidates=[]))


def test_response_without_content_is_an_error() -> None:
    with pytest.raises(LLMError, match="no content"):
        to_response(types.GenerateContentResponse(candidates=[types.Candidate(content=None)]))


def test_user_message_becomes_user_content() -> None:
    (content,) = to_contents([UserMessage("How is a job submitted?")])

    assert content.role == "user"
    assert content.parts[0].text == "How is a job submitted?"


def test_assistant_turn_is_replayed_unchanged() -> None:
    gemini_response = response_with(call_part("grep_code", {"pattern": "Job"}))
    turn = to_response(gemini_response)

    (content,) = to_contents([turn])

    assert content is turn.raw


def test_tool_results_become_one_content_with_a_part_per_call() -> None:
    message = ToolResultsMessage(
        (ToolResult("grep_code", "Job.java:70: class Job"), ToolResult("read_file", "1: package"))
    )

    (content,) = to_contents([message])

    assert content.role == "user"
    assert [part.function_response.name for part in content.parts] == ["grep_code", "read_file"]
    assert content.parts[0].function_response.response == {"result": "Job.java:70: class Job"}


def test_assistant_turn_from_another_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="not produced by GeminiLLM"):
        to_contents([LLMResponse(text="scripted")])


def test_tool_specs_become_function_declarations() -> None:
    spec = ToolSpec(
        "grep_code",
        "Search.",
        {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    )

    tool = to_tool([spec])

    assert [declaration.name for declaration in tool.function_declarations] == ["grep_code"]


def test_generate_disables_automatic_function_calling(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = GeminiLLM(api_key="test-key", model="test-model")
    sent: dict[str, object] = {}

    def fake_generate_content(
        *, model: str, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        sent.update(model=model, config=config)
        return response_with(types.Part(text="done"))

    monkeypatch.setattr(llm._client.models, "generate_content", fake_generate_content)

    result = llm.generate("system prompt", [UserMessage("question")], [])

    config = sent["config"]
    assert result.text == "done"
    assert sent["model"] == "test-model"
    assert config.automatic_function_calling.disable is True
    assert config.system_instruction == "system prompt"
    assert config.tools is None
