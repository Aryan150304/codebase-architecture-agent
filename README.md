# RepoScope

A backend service with an LLM agent that explores an unfamiliar codebase with tools and answers
architecture questions with file- and line-level citations (`path:start-end`).

Benchmark target: the new MapReduce API in Apache Hadoop at tag `rel/release-3.4.1`,
restricted to
`hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-core/src/main/java/org/apache/hadoop/mapreduce`.

Status: v0.1 (walking skeleton) in progress.

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

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run --env-file .env pytest
```

Tests marked `db` need `TEST_DATABASE_URL` and are skipped without it, which is how CI runs them
(`pytest -m "not db"`).
