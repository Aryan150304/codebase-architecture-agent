from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from reposcope.db import replace_chunks
from reposcope.ingestion import Chunk
from reposcope.tools import (
    MAX_GREP_MATCHES,
    MAX_READ_LINES,
    Tools,
    grep_code,
    read_file,
    semantic_search,
)
from tests.fakes import FakeEmbedder

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "minirepo"
GREETER = "src/main/java/com/example/Greeter.java"
JAVA_SUBPATH = "src/main/java"


def test_read_file_numbers_the_requested_lines() -> None:
    result = read_file(FIXTURE_ROOT, GREETER, 3, 4)

    assert result == f"{GREETER}:3-4\n3: public class Greeter {{\n4:     private final String name;"


def test_read_file_stops_at_the_end_of_the_file() -> None:
    result = read_file(FIXTURE_ROOT, GREETER, 12, 999)

    assert result.startswith(f"{GREETER}:12-13")
    assert result.endswith("13: }")


def test_read_file_rejects_a_line_past_the_end() -> None:
    assert read_file(FIXTURE_ROOT, GREETER, 500, 600) == (
        f"error: {GREETER} has 13 lines, so line 500 does not exist"
    )


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0, 5, "error: start_line must be at least 1, got 0"),
        (-1, 5, "error: start_line must be at least 1, got -1"),
        (5, 2, "error: end_line 2 is before start_line 5"),
    ],
)
def test_read_file_rejects_invalid_ranges(start: int, end: int, expected: str) -> None:
    assert read_file(FIXTURE_ROOT, GREETER, start, end) == expected


@pytest.mark.parametrize(
    "path",
    [
        "../../../etc/passwd",
        "src/../../../etc/passwd",
        "/etc/passwd",
        "src/main/java/../../../../outside.txt",
    ],
)
def test_read_file_refuses_to_escape_the_repository(path: str) -> None:
    assert read_file(FIXTURE_ROOT, path, 1, 5) == f"error: {path} is outside the repository"


def test_read_file_rejects_a_directory() -> None:
    assert read_file(FIXTURE_ROOT, "src", 1, 5) == "error: src is not a file in the repository"


def test_read_file_rejects_a_missing_file() -> None:
    assert read_file(FIXTURE_ROOT, "src/Nope.java", 1, 5) == (
        "error: src/Nope.java is not a file in the repository"
    )


def test_read_file_caps_the_number_of_lines(tmp_path: Path) -> None:
    (tmp_path / "Long.java").write_text(
        "\n".join(f"line {number}" for number in range(1, 501)), encoding="utf-8"
    )

    result = read_file(tmp_path, "Long.java", 1, 500)

    assert f"Long.java:1-{MAX_READ_LINES}" in result
    assert f"stopped at line {MAX_READ_LINES}" in result
    assert f"{MAX_READ_LINES + 1}: line {MAX_READ_LINES + 1}" not in result


def test_read_file_follows_symlinks_out_of_the_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive\n", encoding="utf-8")
    (repo / "link.txt").symlink_to(secret)

    assert read_file(repo, "link.txt", 1, 1) == "error: link.txt is outside the repository"


def test_grep_reports_path_line_and_text() -> None:
    result = grep_code(FIXTURE_ROOT, JAVA_SUBPATH, (".java",), r"class Greeter")

    assert result == f"{GREETER}:3: public class Greeter {{"


def test_grep_finds_matches_in_several_files() -> None:
    result = grep_code(FIXTURE_ROOT, JAVA_SUBPATH, (".java",), r"Greeter")

    assert f"{GREETER}:3:" in result
    assert "src/main/java/com/example/Runner.java:5:" in result


def test_grep_only_searches_the_given_extensions() -> None:
    result = grep_code(FIXTURE_ROOT, "", (".java",), r"greeting\.target")

    assert result == "no matches for greeting\\.target"


def test_grep_reports_an_invalid_regular_expression() -> None:
    result = grep_code(FIXTURE_ROOT, JAVA_SUBPATH, (".java",), r"class (Greeter")

    assert result.startswith("error: invalid regular expression:")


def test_grep_reports_a_missing_subpath() -> None:
    result = grep_code(FIXTURE_ROOT, "does-not-exist", (".java",), r"Greeter")

    assert result.startswith("error:")
    assert "is not a directory" in result


def test_grep_caps_the_number_of_matches(tmp_path: Path) -> None:
    (tmp_path / "Many.java").write_text(
        "\n".join(f"match {number}" for number in range(200)), encoding="utf-8"
    )

    result = grep_code(tmp_path, "", (".java",), r"match")

    assert len(result.splitlines()) == MAX_GREP_MATCHES + 1
    assert result.endswith("... more matches exist; narrow the pattern")


@pytest.mark.db
def test_semantic_search_puts_the_closest_chunk_first(session: Session) -> None:
    chunks = [
        Chunk("Job.java", 1, 40, "public class Job implements JobContext"),
        Chunk("Mapper.java", 1, 40, "public class Mapper<KEYIN, VALUEIN>"),
        Chunk("Reducer.java", 1, 40, "public class Reducer<KEYIN, VALUEIN>"),
    ]
    embedder = FakeEmbedder()
    replace_chunks(session, chunks, embedder.encode([chunk.content for chunk in chunks]))

    result = semantic_search(session, embedder, "public class Mapper<KEYIN, VALUEIN>", k=2)

    assert result.startswith("Mapper.java:1-40\n")
    assert len(result.split("\n\n")) == 2


@pytest.mark.db
def test_semantic_search_rejects_an_empty_query(session: Session) -> None:
    assert semantic_search(session, FakeEmbedder(), "   ") == "error: query must not be empty"


@pytest.mark.db
def test_semantic_search_reports_an_empty_index(session: Session) -> None:
    assert semantic_search(session, FakeEmbedder(), "anything") == (
        "no chunks are stored; ingest the repository first"
    )


@pytest.mark.db
def test_semantic_search_clamps_k(session: Session) -> None:
    chunks = [Chunk("A.java", 1, 1, f"content {number}") for number in range(20)]
    embedder = FakeEmbedder()
    replace_chunks(session, chunks, embedder.encode([chunk.content for chunk in chunks]))

    assert len(semantic_search(session, embedder, "content", k=99).split("\n\n")) == 10
    assert len(semantic_search(session, embedder, "content", k=0).split("\n\n")) == 1


@pytest.mark.db
def test_call_dispatches_to_each_tool(session: Session) -> None:
    tools = Tools(FIXTURE_ROOT, JAVA_SUBPATH, (".java",), session, FakeEmbedder())

    assert tools.call("read_file", {"path": GREETER, "start_line": 3, "end_line": 3}).endswith(
        "3: public class Greeter {"
    )
    assert tools.call("grep_code", {"pattern": "class Runner"}).endswith("3: public class Runner {")
    assert tools.call("semantic_search", {"query": "greeter"}).startswith("no chunks are stored")


@pytest.mark.db
def test_call_reports_an_unknown_tool(session: Session) -> None:
    tools = Tools(FIXTURE_ROOT, JAVA_SUBPATH, (".java",), session, FakeEmbedder())

    assert tools.call("list_files", {}) == "error: unknown tool list_files"


@pytest.mark.db
def test_call_reports_wrong_arguments(session: Session) -> None:
    tools = Tools(FIXTURE_ROOT, JAVA_SUBPATH, (".java",), session, FakeEmbedder())

    result = tools.call("read_file", {"path": GREETER, "start": 1})

    assert result.startswith("error: wrong arguments for read_file:")
