"""Embedding model for code chunks."""

from typing import Protocol

from reposcope.db import EMBEDDING_DIM

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"

MAX_SEQ_LENGTH = 2048
"""Tokens per chunk the model is allowed to read.

The longest chunk in the Hadoop MapReduce subtree is 3549 characters, roughly 1200 tokens, so
this leaves headroom without truncating. The model's own default is 8192, which makes attention
memory large enough to exhaust a shared-memory GPU. Raise this if the chunk size grows.
"""

BATCH_SIZE = 8
"""Chunks encoded at once. sentence-transformers sorts longest-first, so the first batch is the
most expensive one and decides whether the run fits in memory."""


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class JinaEmbedder:
    """Local sentence-transformers model, producing unit vectors for cosine distance.

    The model ships custom code for its long context, so it needs `trust_remote_code`.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        show_progress: bool = False,
    ) -> None:
        # Imported here so that importing this module does not pull in torch.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self._model.max_seq_length = MAX_SEQ_LENGTH
        self._batch_size = batch_size
        self._show_progress = show_progress
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension != EMBEDDING_DIM:
            raise RuntimeError(
                f"{model_name} produces {dimension} dimensions, but the schema expects "
                f"{EMBEDDING_DIM}"
            )

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=self._show_progress,
        )
        return [vector.tolist() for vector in vectors]
