"""Turn source files into line-based chunks for embedding."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    path: str
    """Path relative to the repository root, with forward slashes."""
    start_line: int
    """First line of the chunk, 1-based and inclusive."""
    end_line: int
    """Last line of the chunk, inclusive."""
    content: str


def chunk_lines(path: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Split text into windows of `size` lines, each overlapping the previous by `overlap` lines.

    The last window ends at the last line, so it can be shorter than `size`.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got {overlap} with size {size}")

    lines = text.splitlines()
    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    while start < len(lines):
        end = min(start + size, len(lines))
        chunks.append(Chunk(path, start + 1, end, "\n".join(lines[start:end])))
        if end == len(lines):
            break
        start += step
    return chunks
