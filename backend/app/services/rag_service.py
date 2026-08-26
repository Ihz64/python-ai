import math
from typing import List, Optional
from pydantic import BaseModel


class RAGChunk(BaseModel):
    chunk_id: str
    source: str
    content: str
    vector: Optional[List[float]] = None


class InMemoryVectorStore:
    def __init__(self):
        self._chunks: List[RAGChunk] = []

    def _simple_embedding(self, text: str, dim: int = 64) -> List[float]:
        vec = [0.0] * dim
        for i, char in enumerate(text.lower()):
            idx = ord(char) % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def add_document(self, source: str, text: str, chunk_size: int = 500) -> None:
        lines = text.split("\n")
        current_chunk = ""
        chunk_idx = 0

        for line in lines:
            current_chunk += line + "\n"
            if len(current_chunk) >= chunk_size:
                c_id = f"{source}_{chunk_idx}"
                vec = self._simple_embedding(current_chunk)
                self._chunks.append(RAGChunk(chunk_id=c_id, source=source, content=current_chunk, vector=vec))
                current_chunk = ""
                chunk_idx += 1

        if current_chunk.strip():
            c_id = f"{source}_{chunk_idx}"
            vec = self._simple_embedding(current_chunk)
            self._chunks.append(RAGChunk(chunk_id=c_id, source=source, content=current_chunk, vector=vec))

    def similarity_search(self, query: str, top_k: int = 3) -> List[RAGChunk]:
        query_vec = self._simple_embedding(query)

        scored = []
        for chunk in self._chunks:
            if not chunk.vector:
                continue
            dot = sum(q * c for q, c in zip(query_vec, chunk.vector))
            scored.append((dot, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]


vector_store = InMemoryVectorStore()
