import pytest

from reposcope.ingestion import Chunk, chunk_lines


def numbered_lines(count: int) -> str:
    return "\n".join(f"line {i}" for i in range(1, count + 1))


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_lines("a.java", "", size=40, overlap=10) == []


def test_text_shorter_than_size_gives_one_chunk() -> None:
    chunks = chunk_lines("a.java", numbered_lines(5), size=40, overlap=10)

    assert chunks == [Chunk("a.java", 1, 5, numbered_lines(5))]


def test_windows_overlap_and_last_window_ends_at_last_line() -> None:
    chunks = chunk_lines("a.java", numbered_lines(45), size=20, overlap=5)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 20), (16, 35), (31, 45)]


def test_text_exactly_one_window_long_gives_one_chunk() -> None:
    chunks = chunk_lines("a.java", numbered_lines(40), size=40, overlap=10)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 40)]


def test_chunk_content_matches_its_line_range() -> None:
    chunks = chunk_lines("a.java", numbered_lines(30), size=20, overlap=5)

    second = chunks[1]
    assert second.content.splitlines()[0] == "line 16"
    assert second.content.splitlines()[-1] == "line 30"


def test_trailing_newline_does_not_add_a_line() -> None:
    chunks = chunk_lines("a.java", numbered_lines(3) + "\n", size=40, overlap=10)

    assert chunks[0].end_line == 3


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_window_settings_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_lines("a.java", "x", size=size, overlap=overlap)
