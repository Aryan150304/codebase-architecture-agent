import math

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from reposcope.db import EMBEDDING_DIM, ChunkRow, replace_chunks
from reposcope.ingestion import Chunk
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.db


def unit_vector(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = 1.0
    return vector


def diagonal_vector() -> list[float]:
    """Unit vector halfway between axis 0 and axis 1."""
    vector = [0.0] * EMBEDDING_DIM
    vector[0] = vector[1] = 1.0 / math.sqrt(2)
    return vector


def test_chunk_row_round_trips(session: Session) -> None:
    session.add(
        ChunkRow(
            path="org/apache/hadoop/mapreduce/Job.java",
            start_line=1,
            end_line=40,
            content="public class Job {",
            embedding=unit_vector(0),
        )
    )
    session.commit()

    stored = session.scalars(select(ChunkRow)).one()
    assert stored.path == "org/apache/hadoop/mapreduce/Job.java"
    assert (stored.start_line, stored.end_line) == (1, 40)
    assert stored.content == "public class Job {"
    assert len(stored.embedding) == EMBEDDING_DIM


def test_cosine_distance_orders_results_by_similarity(session: Session) -> None:
    session.add_all(
        [
            ChunkRow(
                path="far.java", start_line=1, end_line=1, content="", embedding=unit_vector(1)
            ),
            ChunkRow(
                path="exact.java", start_line=1, end_line=1, content="", embedding=unit_vector(0)
            ),
            ChunkRow(
                path="near.java", start_line=1, end_line=1, content="", embedding=diagonal_vector()
            ),
        ]
    )
    session.commit()

    nearest = session.scalars(
        select(ChunkRow).order_by(ChunkRow.embedding.cosine_distance(unit_vector(0)))
    ).all()

    assert [row.path for row in nearest] == ["exact.java", "near.java", "far.java"]


def test_replace_chunks_is_idempotent(session: Session) -> None:
    chunks = [
        Chunk("Job.java", 1, 40, "public class Job {"),
        Chunk("Job.java", 31, 70, "  public void submit() {"),
    ]
    embeddings = FakeEmbedder().encode([chunk.content for chunk in chunks])

    replace_chunks(session, chunks, embeddings)
    replace_chunks(session, chunks, embeddings)

    assert session.scalar(select(func.count()).select_from(ChunkRow)) == 2


def test_replace_chunks_rejects_mismatched_input_lengths(session: Session) -> None:
    chunks = [Chunk("Job.java", 1, 40, "public class Job {")]

    with pytest.raises(ValueError, match="1 chunks but 0 embeddings"):
        replace_chunks(session, chunks, [])


def test_wrong_embedding_dimension_is_rejected(session: Session) -> None:
    session.add(
        ChunkRow(path="a.java", start_line=1, end_line=1, content="", embedding=[1.0, 0.0, 0.0])
    )

    with pytest.raises(DBAPIError):
        session.commit()
