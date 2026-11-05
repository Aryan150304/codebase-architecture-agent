"""Chunk storage: schema, engine and schema creation."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from reposcope.config import load_settings

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


def main() -> None:
    engine = create_db_engine(load_settings().database_url)
    create_schema(engine)
    print(f"schema ready in {engine.url.render_as_string(hide_password=True)}")


if __name__ == "__main__":
    main()
