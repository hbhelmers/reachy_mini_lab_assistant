from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

CHUNK_MIN_CHARS = 100
SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass
class Chunk:
    text: str
    source_file: str
    category: str
    chunk_index: int


def iter_content_files(content_dir: str) -> list[Path]:
    """Return supported content files under a directory, recursively."""
    root = Path(content_dir)
    if not root.exists():
        logger.warning("Content directory does not exist: %s", content_dir)
        return []

    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("_"):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files.append(path)

    return sorted(files)


def read_file(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def chunk_text(content: str) -> list[str]:
    """Chunk markdown/text by headers or paragraph breaks."""
    text = content.strip()
    if not text:
        return []

    # Prefer markdown-style section splits when possible
    if "\n#" in text or text.startswith("#"):
        sections = re.split(r"\n(?=#{1,3}\s)", text)
    else:
        # Fallback: split on blank lines
        sections = re.split(r"\n\s*\n+", text)

    chunks = [section.strip() for section in sections if len(section.strip()) >= CHUNK_MIN_CHARS]

    if not chunks:
        chunks = [text]

    return chunks


def category_from_path(file_path: Path, root_dir: Path) -> str:
    """Derive category from the first directory under root, else 'general'."""
    try:
        relative = file_path.relative_to(root_dir)
    except ValueError:
        return "general"

    parts = relative.parts
    if len(parts) >= 2:
        return parts[0]
    return "general"


def build_chunks(content_dir: str) -> list[Chunk]:
    """Load files from a local content directory and return chunk objects."""
    root = Path(content_dir)
    chunks: list[Chunk] = []

    for file_path in iter_content_files(content_dir):
        try:
            content = read_file(file_path)
            texts = chunk_text(content)
            category = category_from_path(file_path, root)
            source_file = str(file_path.relative_to(root)).replace("\\", "/")

            for i, text in enumerate(texts):
                chunks.append(
                    Chunk(
                        text=text,
                        source_file=source_file,
                        category=category,
                        chunk_index=i,
                    )
                )

            logger.debug("Chunked %s -> %d chunks", source_file, len(texts))
        except Exception:
            logger.warning("Failed to load/chunk %s", file_path, exc_info=True)

    return chunks