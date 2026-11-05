"""Settings read from the environment."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str
    repo_root: Path
    """Checkout the tools read from. Paths stored in the database are relative to it."""
    repo_subpath: str
    """Sub-directory the tools serve, which must match what was ingested."""
    repo_extensions: tuple[str, ...]


def load_settings() -> Settings:
    return Settings(
        database_url=require_env("DATABASE_URL"),
        repo_root=Path(require_env("REPO_ROOT")).expanduser().resolve(),
        repo_subpath=os.environ.get("REPO_SUBPATH", ""),
        repo_extensions=parse_extensions(require_env("REPO_EXTENSIONS")),
    )


def parse_extensions(value: str) -> tuple[str, ...]:
    extensions = tuple(item.strip() for item in value.split(",") if item.strip())
    if not extensions:
        raise RuntimeError("REPO_EXTENSIONS must list at least one file suffix")
    return extensions


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set")
    return value
