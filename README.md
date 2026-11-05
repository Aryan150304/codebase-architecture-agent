# RepoScope

A backend service with an LLM agent that explores an unfamiliar codebase with tools and answers
architecture questions with file- and line-level citations (`path:start-end`).

It continues a university project on architecture recovery of Apache Hadoop MapReduce, making
that process agentic and interactive rather than a batch pipeline.

Benchmark target: the new MapReduce API in Apache Hadoop at tag `rel/release-3.4.1`, restricted
to
`hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-core/src/main/java/org/apache/hadoop/mapreduce`
— 332 Java files.

Status: v0.1 (walking skeleton) in progress. Ingestion works; tools, agent, API and eval are next.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- PostgreSQL with the `pgvector` extension (developed against 18.6 from Postgres.app)

## Setup

```bash
cp .env.example .env
createdb reposcope
createdb reposcope_test
psql reposcope -c "CREATE EXTENSION vector;"
psql reposcope_test -c "CREATE EXTENSION vector;"

uv sync
uv run --env-file .env python -m reposcope.db
```

## Ingestion

Walks a local checkout, splits each file into overlapping windows of lines, embeds them with
`jinaai/jina-embeddings-v2-base-code`, and replaces the contents of the `chunks` table. Each
chunk keeps its exact line range, so a retrieval hit maps directly to a `path:start-end`
citation. The first run downloads about 320 MB of model weights.

```bash
uv run --env-file .env reposcope ingest ~/Desktop/hadoop \
  --subpath hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-core/src/main/java/org/apache/hadoop/mapreduce \
  --ext .java
```

Re-running replaces every row, so ingestion is repeatable rather than cumulative.

Measured on an Apple M1 with 8 GB of shared memory, at the default 40-line chunks with 10 lines
of overlap: 2141 chunks from 332 files, embedded in 3m15s. If the GPU runs out of memory, lower
`--batch-size`; see `docs/devlog.md` for why the defaults are what they are.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run --env-file .env pytest
```

Tests marked `db` need `TEST_DATABASE_URL` and are skipped without it, which is how CI runs them
(`pytest -m "not db"`). Tests never download the embedding model or call an API: `tests/fakes.py`
provides deterministic stand-ins, and `tests/fixtures/minirepo` is a tiny hand-written repository.

## Known limitations

- The embedding model is not pinned to a revision, so Hugging Face serves whatever is current.
  Eval numbers are not fully reproducible until it is.
- `transformers` is pinned below version 5 because the model's remote code requires it.
- Chunks are fixed line windows, so they cut through methods and classes. Symbol-aware chunking
  is a candidate for a later version, triggered by weak file recall in the eval.
- No vector index. An exact scan over 2141 rows is fast; this needs revisiting at a larger scale.
- Ingestion is all-or-nothing and single-threaded: there is no incremental update.
