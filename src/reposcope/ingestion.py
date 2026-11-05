"""Turn a repository's source files into line-based chunks for embedding."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    path: str
    """Path relative to the repository root, with forward slashes."""
    start_line: int
    """First line of the chunk, 1-based and inclusive."""
    end_line: int
    """Last line of the chunk, inclusive."""
    content: str


@dataclass(frozen=True)
class RepoChunks:
    chunks: list[Chunk]
    files: int
    skipped: list[Path]
    """Files that could not be decoded as UTF-8 text."""


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


def iter_source_files(root: Path, subpath: str, extensions: tuple[str, ...]) -> Iterator[Path]:
    """Yield files under `root / subpath` with a matching suffix, in a stable order."""
    base = root / subpath
    if not base.is_dir():
        raise ValueError(f"{base} is not a directory")
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix in extensions:
            yield path


def chunk_repo(
    root: Path,
    subpath: str,
    extensions: tuple[str, ...],
    size: int,
    overlap: int,
) -> RepoChunks:
    """Chunk every matching file, reporting files that had to be skipped."""
    chunks: list[Chunk] = []
    skipped: list[Path] = []
    files = 0
    for path in iter_source_files(root, subpath, extensions):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            skipped.append(path)
            continue
        files += 1
        chunks.extend(chunk_lines(path.relative_to(root).as_posix(), text, size, overlap))
    return RepoChunks(chunks=chunks, files=files, skipped=skipped)
