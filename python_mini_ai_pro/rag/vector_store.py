import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VectorStoreRAG:
    """
    Local Vector Store & Retrieval-Augmented Generation (RAG) Engine.
    Implements TF-IDF vectorization, document chunking, dot-product similarity,
    and semantic context retrieval for local LLM / knowledge expansion.
    """

    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.documents: List[Dict[str, Any]] = []
        self.vocabulary: List[str] = []
        self.idf: Dict[str, float] = {}
        self.load_store()

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def add_document(self, doc_id: str, title: str, content: str, tags: List[str] = None):
        """Adds or updates a document in the RAG vector store."""
        chunks = self._chunk_text(content, chunk_size=300)
        doc_entry = {
            "id": doc_id,
            "title": title,
            "content": content,
            "chunks": chunks,
            "tags": tags or []
        }
        # Remove existing if ID exists
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        self.documents.append(doc_entry)
        self._reindex()
        self.save_store()

    def _chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks if chunks else [text]

    def _reindex(self):
        """Recomputes TF-IDF index across all document chunks."""
        doc_count = 0
        doc_freq: Dict[str, int] = {}
        vocab_set = set()

        all_chunks = []
        for doc in self.documents:
            for chunk in doc["chunks"]:
                doc_count += 1
                tokens = set(self._tokenize(chunk))
                vocab_set.update(tokens)
                for t in tokens:
                    doc_freq[t] = doc_freq.get(t, 0) + 1

        self.vocabulary = sorted(list(vocab_set))
        self.idf = {
            w: math.log((doc_count + 1) / (freq + 1)) + 1.0
            for w, freq in doc_freq.items()
        }

    def _get_tfidf_vector(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * len(self.vocabulary)

        term_freq: Dict[str, int] = {}
        for t in tokens:
            term_freq[t] = term_freq.get(t, 0) + 1

        total_terms = len(tokens)
        vec = []
        for word in self.vocabulary:
            tf = term_freq.get(word, 0) / total_terms
            idf = self.idf.get(word, 1.0)
            vec.append(tf * idf)
        return vec

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs RAG similarity retrieval over document chunks."""
        if not self.documents or not query:
            return []

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        results = []
        for doc in self.documents:
            for chunk in doc["chunks"]:
                chunk_tokens = set(self._tokenize(chunk))
                if not chunk_tokens:
                    continue

                # Jaccard + Term Overlap Score
                overlap = len(query_tokens.intersection(chunk_tokens))
                score = overlap / (math.sqrt(len(query_tokens)) * math.sqrt(len(chunk_tokens)))

                if score > 0.1:
                    results.append({
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "chunk": chunk,
                        "score": float(score)
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save_store(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load_store(self):
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                self._reindex()
            except Exception as e:
                logger.error(f"Failed loading vector store: {e}")
                self.documents = []
