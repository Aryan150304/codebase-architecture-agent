"""Provider-neutral vocabulary for talking to a language model.

The agent speaks only these types, so it never imports an SDK and tests can script the model.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class LLMError(RuntimeError):
    """The model could not produce a usable response."""


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call, with a JSON Schema describing its arguments."""

    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class ToolResultsMessage:
    """Results for every tool call of one model turn, in the order they were requested."""

    results: tuple[ToolResult, ...]


@dataclass(frozen=True)
class LLMResponse:
    """One model turn: final text, tool calls, or both.

    `raw` is the provider's own form of the turn. It goes back into the history unchanged,
    because providers attach data that has to survive the round trip.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    raw: object = None


type Message = UserMessage | LLMResponse | ToolResultsMessage


class LLM(Protocol):
    def generate(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> LLMResponse: ...
