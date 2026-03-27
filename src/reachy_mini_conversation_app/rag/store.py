from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams


logger = logging.getLogger(__name__)

COLLECTION_NAME = "lab_content"
VECTOR_SIZE = 1536  # text-embedding-3-small


class VectorStore:
    """Local Qdrant-backed vector store for lab assistant content."""

    def __init__(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)

    def is_empty(self) -> bool:
        info = self._client.get_collection(COLLECTION_NAME)
        return info.points_count == 0

    def upsert(self, points: list[PointStruct]) -> None:
        if not points:
            return
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)

    def delete_by_file(self, file_path: str) -> None:
        self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=file_path))]
            ),
        )

    def search(
        self,
        query_vector: list[float],
        category: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if category:
            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "text": point.payload.get("text", ""),
                "source": point.payload.get("source_file", ""),
                "category": point.payload.get("category", "general"),
                "score": point.score,
            }
            for point in response.points
        ]