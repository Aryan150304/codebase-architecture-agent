"""Command line interface."""

import argparse
from pathlib import Path

from sqlalchemy.orm import Session

from reposcope.config import load_settings
from reposcope.db import create_db_engine, create_schema, replace_chunks
from reposcope.embedding import BATCH_SIZE, JinaEmbedder
from reposcope.ingestion import chunk_repo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reposcope")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser(
        "ingest",
        help="embed the configured repository into the chunks table",
        description=(
            "Reads REPO_ROOT, REPO_SUBPATH and REPO_EXTENSIONS from the environment. "
            "The arguments below override them for one run."
        ),
    )
    ingest.add_argument("repo_root", type=Path, nargs="?")
    ingest.add_argument("--subpath", help="restrict ingestion to this sub-directory")
    ingest.add_argument(
        "--ext",
        dest="extensions",
        action="append",
        help="file suffix to ingest, including the dot; repeat for several",
    )
    ingest.add_argument("--chunk-size", type=int, default=40)
    ingest.add_argument("--chunk-overlap", type=int, default=10)
    ingest.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="chunks embedded at once; lower it if the GPU runs out of memory",
    )
    ingest.set_defaults(handler=run_ingest)

    return parser


def run_ingest(args: argparse.Namespace) -> None:
    settings = load_settings()
    repo_root = (
        settings.repo_root if args.repo_root is None else args.repo_root.expanduser().resolve()
    )
    subpath = settings.repo_subpath if args.subpath is None else args.subpath
    extensions = tuple(args.extensions) if args.extensions else settings.repo_extensions

    result = chunk_repo(repo_root, subpath, extensions, args.chunk_size, args.chunk_overlap)
    print(f"{len(result.chunks)} chunks from {result.files} files under {repo_root / subpath}")
    for path in result.skipped:
        print(f"skipped (not UTF-8 text): {path}")
    if not result.chunks:
        return

    embedder = JinaEmbedder(batch_size=args.batch_size, show_progress=True)
    embeddings = embedder.encode([chunk.content for chunk in result.chunks])

    engine = create_db_engine(settings.database_url)
    create_schema(engine)
    with Session(engine) as session:
        stored = replace_chunks(session, result.chunks, embeddings)
    print(f"stored {stored} chunks")


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
