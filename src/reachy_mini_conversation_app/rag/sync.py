from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

from qdrant_client.models import PointStruct

from reachy_mini_conversation_app.rag.embeddings import Embeddings
from reachy_mini_conversation_app.rag.loader import build_chunks, iter_content_files
from reachy_mini_conversation_app.rag.store import VectorStore


logger = logging.getLogger(__name__)


def _point_id(source_file: str, chunk_index: int) -> str:
    """Stable deterministic ID for a chunk."""
    raw = f"{source_file}:{chunk_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    """Hash file contents for change detection."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContentSyncWorker(threading.Thread):
    """Sync local content directory into vector store once at startup."""

    def __init__(
        self,
        content_dir: str,
        store: VectorStore,
        embeddings: Embeddings,
        state_path: str,
    ) -> None:
        super().__init__(name="ContentSyncWorker", daemon=True)
        self._content_dir = Path(content_dir)
        self._store = store
        self._embeddings = embeddings
        self._state_path = Path(state_path)
        self.ready = threading.Event()
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self._sync()
        except Exception as e:
            logger.error("ContentSyncWorker failed: %s", e, exc_info=True)
            self.error = e
        finally:
            self.ready.set()

    def _load_state(self) -> dict:
        if self._state_path.exists():
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        return {"files": {}}

    def _save_state(self, state: dict) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _current_file_map(self) -> dict[str, str]:
        files: dict[str, str] = {}
        for path in iter_content_files(str(self._content_dir)):
            rel = str(path.relative_to(self._content_dir)).replace("\\", "/")
            files[rel] = _file_sha(path)
        return files

    def _sync(self) -> None:
        logger.info("ContentSyncWorker: checking local content in %s", self._content_dir)

        current_files = self._current_file_map()
        state = self._load_state()
        stored_files: dict[str, str] = state.get("files", {})

        changed = [path for path, sha in current_files.items() if stored_files.get(path) != sha]
        removed = [path for path in stored_files if path not in current_files]

        if not changed and not removed:
            logger.info("ContentSyncWorker: content up to date, nothing to do")
            return

        logger.info(
            "ContentSyncWorker: %d changed, %d removed files",
            len(changed),
            len(removed),
        )

        # Delete old chunks for removed or changed files
        for path in removed + changed:
            self._store.delete_by_file(path)

        # Rebuild embeddings for all current content, then keep only changed paths
        if changed:
            all_chunks = build_chunks(str(self._content_dir))
            chunks = [chunk for chunk in all_chunks if chunk.source_file in changed]

            texts = [chunk.text for chunk in chunks]
            vectors = self._embeddings.embed(texts)

            points = [
                PointStruct(
                    id=_point_id(chunk.source_file, chunk.chunk_index),
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "category": chunk.category,
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ]

            self._store.upsert(points)
            logger.info("ContentSyncWorker: upserted %d chunks", len(points))

        new_files = {**stored_files, **{path: current_files[path] for path in changed}}
        for path in removed:
            new_files.pop(path, None)

        self._save_state({"files": new_files})
        logger.info("ContentSyncWorker: sync complete")