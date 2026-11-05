"""Test doubles that keep tests offline."""

import hashlib
import math
import random

from reposcope.db import EMBEDDING_DIM


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
