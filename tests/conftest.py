import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, delete
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from reposcope.db import ChunkRow, create_db_engine, create_schema


@pytest.fixture(scope="session")
def engine() -> Engine:
    """Engine for the throwaway test database, skipping if it is not reachable."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    test_engine = create_db_engine(url)
    try:
        create_schema(test_engine)
    except OperationalError as error:
        pytest.skip(f"test database is not reachable: {error}")
    return test_engine


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Session against an empty chunks table."""
    with Session(engine) as open_session:
        open_session.execute(delete(ChunkRow))
        open_session.commit()
        yield open_session
