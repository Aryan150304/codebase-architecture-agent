"""Test doubles that keep tests offline."""

import hashlib
import math
import random
from collections.abc import Sequence

from reposcope.db import EMBEDDING_DIM
from reposcope.llm import LLMResponse, Message, ToolSpec


class FakeEmbedder:
    """Deterministic embeddings seeded from the text, so no model has to be downloaded."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).hexdigest()
        generator = random.Random(seed)
        raw = [generator.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)]
        norm = math.sqrt(sum(value * value for value in raw))
        return [value / norm for value in raw]


class FakeLLM:
    """Returns scripted responses in order and records what the agent sent each time."""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._remaining = list(responses)
        self.requests: list[list[Message]] = []

    def generate(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        self.requests.append(list(messages))
        if not self._remaining:
            raise AssertionError("FakeLLM was called more often than it has scripted responses")
        return self._remaining.pop(0)
