from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Iterable, List

from src.config import Settings

logging.basicConfig(level=logging.INFO)


try:
    SET = Settings.model_validate_yaml(Path("settings.yaml"))
except Exception:  # pragma: no cover - fallback to defaults
    SET = Settings()


def _gather_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for file in path.rglob("*"):
                if file.is_file():
                    files.append(file)
        elif path.is_file():
            files.append(path)
        else:
            logging.warning("Skipping unknown path %s", path)
    return files


def _load_db(path: str) -> object:
    module = importlib.import_module("documentdb")
    DocumentDB = getattr(module, "DocumentDB")
    return DocumentDB(path)


def _add(paths: Iterable[str], docstore_path: str) -> None:
    db = _load_db(docstore_path)
    files = _gather_files(paths)
    documents = []
    for file in files:
        try:
            documents.append({"id": str(file), "text": file.read_text(encoding="utf-8")})
        except Exception as exc:  # pragma: no cover - logging only
            logging.error("Failed to read %s: %s", file, exc)
    if documents:
        db.add_documents(documents)
    logging.info("Indexed %d documents", len(documents))


def _remove(paths: Iterable[str], docstore_path: str) -> None:
    db = _load_db(docstore_path)
    files = _gather_files(paths)
    ids = [str(file) for file in files]
    if hasattr(db, "delete_documents"):
        db.delete_documents(ids)
    elif hasattr(db, "remove_documents"):
        db.remove_documents(ids)
    else:  # pragma: no cover - defensive branch
        raise AttributeError("DocumentDB does not support removing documents")
    logging.info("Removed %d documents", len(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest or remove documents")
    parser.add_argument(
        "--path",
        default=SET.docstore_path,
        help="Path to document store (default: settings.yaml docstore_path)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--add", nargs="+", help="Paths of files or directories to index")
    group.add_argument("--remove", nargs="+", help="Paths of files or directories to remove")
    args = parser.parse_args()

    if args.add:
        _add(args.add, args.path)
    elif args.remove:
        _remove(args.remove, args.path)


if __name__ == "__main__":
    main()
