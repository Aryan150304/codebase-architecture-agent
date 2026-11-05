"""Gemini implementation of the LLM protocol, using native function calling."""

from collections.abc import Mapping, Sequence

from google import genai
from google.genai import errors, types

from reposcope.llm import (
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)


class GeminiLLM:
    """Gemini with automatic function calling switched off.

    The SDK can run the whole tool loop itself. The agent runs it instead, so that every step is
    counted and bounded and the eval can see what the model did.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[to_tool(tools)] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=to_contents(messages),
                config=config,
            )
        except errors.APIError as error:
            raise LLMError(f"gemini request failed with {error.code}: {error.message}") from error
        return to_response(response)


def to_tool(tools: Sequence[ToolSpec]) -> types.Tool:
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.parameters,
            )
            for tool in tools
        ]
    )


def to_contents(messages: Sequence[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in messages:
        match message:
            case UserMessage(text=text):
                contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
            case LLMResponse(raw=types.Content() as raw):
                contents.append(raw)
            case LLMResponse():
                raise ValueError("assistant message was not produced by GeminiLLM")
            case ToolResultsMessage(results=results):
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=result.name,
                                    response={"result": result.content},
                                )
                            )
                            for result in results
                        ],
                    )
                )
            case _:
                raise TypeError(f"unsupported message type {type(message).__name__}")
    return contents


def to_response(response: types.GenerateContentResponse) -> LLMResponse:
    if not response.candidates:
        raise LLMError(
            f"gemini returned no candidates (prompt feedback: {response.prompt_feedback})"
        )
    candidate = response.candidates[0]
    content = candidate.content
    if content is None or not content.parts:
        raise LLMError(f"gemini returned no content (finish reason: {candidate.finish_reason})")

    text = "".join(part.text for part in content.parts if part.text and not part.thought)
    tool_calls = tuple(
        ToolCall(part.function_call.name, normalise_arguments(part.function_call.args or {}))
        for part in content.parts
        if part.function_call and part.function_call.name
    )
    return LLMResponse(text=text, tool_calls=tool_calls, raw=content)


def normalise_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Turn whole-number floats into ints.

    JSON has a single number type, so a decoder that produces doubles hands the tools 10.0 where
    they expect 10, and `range(10.0)` raises.
    """
    return {
        name: int(value) if isinstance(value, float) and value.is_integer() else value
        for name, value in arguments.items()
    }
