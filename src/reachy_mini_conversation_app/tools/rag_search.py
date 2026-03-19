from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

INDEX_DIR = Path("rag/index")

class RagSearch(Tool):
    name = "rag_search"
    description = "Søker i kunnskapsbasen for informasjon om labutstyr, utlån, regler og prosedyrer."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Spørsmålet som skal søkes etter"},
            "top_k": {"type": "integer", "default": 3}
        },
        "required": ["query"]
    }

    def __init__(self):
        self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        self.metadata = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    async def __call__(self, deps: ToolDependencies, **kwargs):
        query = kwargs["query"]
        top_k = int(kwargs.get("top_k", 3))

        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.metadata[idx]
            results.append({
                "source": item["source"],
                "score": float(score),
                "text": item["text"]
            })

        if not results:
            return {"results": [], "message": "Fant ingen relevant informasjon."}

        return {"results": results}