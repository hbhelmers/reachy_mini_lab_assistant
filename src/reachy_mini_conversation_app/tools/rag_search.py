import logging
from typing import Any, Dict
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

class RagSearch(Tool):
    name = "rag_search"
    description = "Search the lab knowledge base for relevant information."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to look up in the knowledge base."
            },
            "category": {
                "type": "string",
                "description": "Optional category filter."
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        query = kwargs.get("query", "").strip()
        category = kwargs.get("category") or None

        if not query:
            return {"error": "query is required"}

        if getattr(deps, "vector_store", None) is None or getattr(deps, "embeddings", None) is None:
            return {"error": "RAG pipeline not available"}

        try:
            query_vector = deps.embeddings.embed_one(query)
            results = deps.vector_store.search(query_vector, category=category, limit=3)

            if not results:
                return {"answer": "Jeg finner ikke dette i dokumentasjonen."}

            context = "\n\n---\n\n".join(
                f"[{r['source']}]\n{r['text']}" for r in results
            )
            return {"context": context}

        except Exception as e:
            logger.error("rag_search failed: %s", e, exc_info=True)
            return {"error": f"Knowledge base lookup failed: {e}"}