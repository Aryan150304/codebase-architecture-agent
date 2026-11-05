from pathlib import Path

import pytest

from reposcope.ingestion import chunk_repo, iter_source_files

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "minirepo"
JAVA_SUBPATH = "src/main/java"


def test_walker_finds_only_matching_extensions() -> None:
    found = list(iter_source_files(FIXTURE_ROOT, "", (".java",)))

    assert [path.name for path in found] == ["Greeter.java", "Runner.java"]


def test_walker_restricts_to_the_subpath() -> None:
    found = list(iter_source_files(FIXTURE_ROOT, "docs", (".md",)))

    assert [path.name for path in found] == ["notes.md"]


def test_walker_accepts_several_extensions() -> None:
    found = list(iter_source_files(FIXTURE_ROOT, "", (".java", ".properties")))

    assert [path.name for path in found] == ["Greeter.java", "Runner.java", "app.properties"]


def test_walker_rejects_a_missing_subpath() -> None:
    with pytest.raises(ValueError, match="is not a directory"):
        list(iter_source_files(FIXTURE_ROOT, "does-not-exist", (".java",)))


def test_chunk_repo_uses_paths_relative_to_the_root() -> None:
    result = chunk_repo(FIXTURE_ROOT, JAVA_SUBPATH, (".java",), size=100, overlap=10)

    assert sorted({chunk.path for chunk in result.chunks}) == [
        "src/main/java/com/example/Greeter.java",
        "src/main/java/com/example/Runner.java",
    ]


def test_chunk_repo_reports_file_count_and_line_ranges() -> None:
    result = chunk_repo(FIXTURE_ROOT, JAVA_SUBPATH, (".java",), size=100, overlap=10)

    assert result.files == 2
    assert result.skipped == []
    ranges = {
        chunk.path.rsplit("/", 1)[-1]: (chunk.start_line, chunk.end_line) for chunk in result.chunks
    }
    assert ranges == {"Greeter.java": (1, 13), "Runner.java": (1, 8)}


def test_chunk_repo_splits_a_file_into_several_chunks() -> None:
    result = chunk_repo(FIXTURE_ROOT, JAVA_SUBPATH, (".java",), size=5, overlap=1)

    greeter = [chunk for chunk in result.chunks if chunk.path.endswith("Greeter.java")]
    assert [(chunk.start_line, chunk.end_line) for chunk in greeter] == [(1, 5), (5, 9), (9, 13)]


def test_chunk_repo_skips_files_that_are_not_utf8(tmp_path: Path) -> None:
    (tmp_path / "ok.java").write_text("class Ok {}\n", encoding="utf-8")
    broken = tmp_path / "broken.java"
    broken.write_bytes(b"class Broken { // \xff\xfe\n}\n")

    result = chunk_repo(tmp_path, "", (".java",), size=40, overlap=10)

    assert result.files == 1
    assert result.skipped == [broken]
    assert [chunk.path for chunk in result.chunks] == ["ok.java"]
