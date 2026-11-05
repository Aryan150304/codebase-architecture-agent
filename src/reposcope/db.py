"""Chunk storage: schema, engine and schema creation."""

from collections.abc import Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine, Integer, String, Text, create_engine, delete, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from reposcope.config import load_settings
from reposcope.ingestion import Chunk

EMBEDDING_DIM = 768
"""Output dimension of jinaai/jina-embeddings-v2-base-code."""


class Base(DeclarativeBase):
    pass


class ChunkRow(Base):
    """A stored chunk. Mirrors `ingestion.Chunk` plus its embedding and row id."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(1024))
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


def create_db_engine(database_url: str) -> Engine:
    return create_engine(database_url)


def create_schema(engine: Engine) -> None:
    """Create the pgvector extension and the chunks table. Safe to call repeatedly."""
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def replace_chunks(
    session: Session,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """Replace every stored chunk, so re-ingesting a repository does not duplicate rows."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"got {len(chunks)} chunks but {len(embeddings)} embeddings")
    session.execute(delete(ChunkRow))
    session.add_all(
        [
            ChunkRow(
                path=chunk.path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                embedding=list(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
    )
    session.commit()
    return len(chunks)


def main() -> None:
    engine = create_db_engine(load_settings().database_url)
    create_schema(engine)
    print(f"schema ready in {engine.url.render_as_string(hide_password=True)}")


if __name__ == "__main__":
    main()
