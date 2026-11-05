"""Tools the agent calls to explore the repository.

Every tool returns text for the model to read, reports failures as text instead of raising, and
caps its output so that one call cannot fill the context window.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from reposcope.db import ChunkRow
from reposcope.embedding import Embedder
from reposcope.ingestion import iter_source_files
from reposcope.llm import ToolSpec

DEFAULT_K = 5
MAX_K = 10
MAX_RESULT_CHARS = 8000
MAX_SNIPPET_CHARS = 1500
MAX_READ_LINES = 200
MAX_GREP_MATCHES = 50

TOOL_SPECS = (
    ToolSpec(
        name="semantic_search",
        description=(
            "Find code by meaning. Describe what you are looking for in natural language and get "
            "the closest snippets, each headed by path:start-end. Good for locating where a "
            "concept is handled. Weak at finding an exact name, which grep_code does better."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in natural language.",
                },
                "k": {
                    "type": "integer",
                    "description": f"Snippets to return, 1 to {MAX_K}. Defaults to {DEFAULT_K}.",
                },
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="read_file",
        description=(
            "Read lines of a source file. Paths are relative to the repository root, exactly as "
            f"they appear in search results. Returns numbered lines, at most {MAX_READ_LINES} per "
            "call, so read long files in ranges."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the repository."},
                "start_line": {"type": "integer", "description": "First line, counting from 1."},
                "end_line": {"type": "integer", "description": "Last line, inclusive."},
            },
            "required": ["path", "start_line", "end_line"],
        },
    ),
    ToolSpec(
        name="grep_code",
        description=(
            "Search every source file for lines matching a regular expression (Python re "
            f"syntax). Returns path:line: text for each match, up to {MAX_GREP_MATCHES}. Good for "
            "exact names such as a class or method. Narrow the pattern if results are cut off."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
            },
            "required": ["pattern"],
        },
    ),
)


def semantic_search(session: Session, embedder: Embedder, query: str, k: int = DEFAULT_K) -> str:
    """Return the chunks closest to `query`, each headed by `path:start-end`."""
    if not query.strip():
        return "error: query must not be empty"
    embedding = embedder.encode([query])[0]
    rows = session.scalars(
        select(ChunkRow)
        .order_by(ChunkRow.embedding.cosine_distance(embedding))
        .limit(min(max(k, 1), MAX_K))
    ).all()
    if not rows:
        return "no chunks are stored; ingest the repository first"
    blocks = [
        f"{row.path}:{row.start_line}-{row.end_line}\n{_cap(row.content, MAX_SNIPPET_CHARS)}"
        for row in rows
    ]
    return _cap("\n\n".join(blocks))


def read_file(repo_root: Path, path: str, start_line: int, end_line: int) -> str:
    """Return the requested lines of a repository file, each prefixed with its line number."""
    if start_line < 1:
        return f"error: start_line must be at least 1, got {start_line}"
    if end_line < start_line:
        return f"error: end_line {end_line} is before start_line {start_line}"

    resolved = _resolve_in_repo(repo_root, path)
    if resolved is None:
        return f"error: {path} is outside the repository"
    if not resolved.is_file():
        return f"error: {path} is not a file in the repository"
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return f"error: cannot read {path}: {error}"

    lines = text.splitlines()
    if start_line > len(lines):
        return f"error: {path} has {len(lines)} lines, so line {start_line} does not exist"

    last = min(end_line, len(lines), start_line + MAX_READ_LINES - 1)
    numbered = "\n".join(f"{number}: {lines[number - 1]}" for number in range(start_line, last + 1))
    body = f"{path}:{start_line}-{last}\n{numbered}"
    if last < min(end_line, len(lines)):
        body += f"\n... stopped at line {last}; read on from there to see more"
    return _cap(body)


def grep_code(repo_root: Path, subpath: str, extensions: tuple[str, ...], pattern: str) -> str:
    """Return `path:line: text` for lines matching `pattern` within the served subtree."""
    try:
        regex = re.compile(pattern)
    except re.error as error:
        return f"error: invalid regular expression: {error}"

    try:
        found = list(
            islice(_iter_matches(repo_root, subpath, extensions, regex), MAX_GREP_MATCHES + 1)
        )
    except ValueError as error:
        return f"error: {error}"

    if not found:
        return f"no matches for {pattern}"
    if len(found) > MAX_GREP_MATCHES:
        shown = "\n".join(found[:MAX_GREP_MATCHES])
        return _cap(f"{shown}\n... more matches exist; narrow the pattern")
    return _cap("\n".join(found))


@dataclass(frozen=True)
class Tools:
    """Binds the tools to one repository and database session, and dispatches by name."""

    repo_root: Path
    repo_subpath: str
    repo_extensions: tuple[str, ...]
    session: Session
    embedder: Embedder

    def call(self, name: str, arguments: dict[str, object]) -> str:
        try:
            if name == "semantic_search":
                return semantic_search(self.session, self.embedder, **arguments)  # type: ignore[arg-type]
            if name == "read_file":
                return read_file(self.repo_root, **arguments)  # type: ignore[arg-type]
            if name == "grep_code":
                return grep_code(
                    self.repo_root,
                    self.repo_subpath,
                    self.repo_extensions,
                    **arguments,  # type: ignore[arg-type]
                )
        except TypeError as error:
            return f"error: wrong arguments for {name}: {error}"
        return f"error: unknown tool {name}"


def _resolve_in_repo(repo_root: Path, path: str) -> Path | None:
    """Resolve a repository-relative path, returning None if it escapes the root.

    Resolution follows symlinks and `..`, so a link pointing outside the repository is rejected
    just like an explicit traversal. An absolute path replaces the root and is rejected too.
    """
    root = repo_root.resolve()
    candidate = (root / path).resolve()
    return candidate if candidate.is_relative_to(root) else None


def _iter_matches(
    repo_root: Path,
    subpath: str,
    extensions: tuple[str, ...],
    regex: re.Pattern[str],
) -> Iterator[str]:
    for file_path in iter_source_files(repo_root, subpath, extensions):
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = file_path.relative_to(repo_root).as_posix()
        for number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                yield f"{relative}:{number}: {line.strip()}"


def _cap(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... truncated at {limit} characters"
