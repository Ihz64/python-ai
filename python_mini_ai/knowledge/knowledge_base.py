import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from python_mini_ai.nlp.similarity import SimilarityCalculator

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Manages local JSON knowledge database with similarity search."""

    def __init__(self, knowledge_file: Path):
        self.knowledge_file = Path(knowledge_file)
        self.similarity_calculator = SimilarityCalculator()
        self.entries: List[Dict[str, Any]] = []
        self.load_knowledge()

    def get_default_knowledge(self) -> List[Dict[str, Any]]:
        return [
            {
                "topic": "Python",
                "question": "Was ist Python?",
                "answer": "Python ist eine vielseitige, übersichtliche und weit verbreitete High-Level-Programmiersprache.",
                "tags": ["python", "programmierung", "sprache"]
            },
            {
                "topic": "Neuronale Netze",
                "question": "Was ist ein neuronales Netzwerk?",
                "answer": "Ein neuronales Netzwerk ist ein mathematisches Modell der KI, das vom menschlichen Gehirn inspiriert ist und aus verknüpften Neuronen besteht.",
                "tags": ["ki", "neural", "netzwerk", "deep learning"]
            },
            {
                "topic": "Machine Learning",
                "question": "Was ist Machine Learning?",
                "answer": "Machine Learning (Maschinelles Lernen) bezeichnet Verfahren, bei denen Computer aus Daten Muster lernen, ohne explizit dafür programmiert zu werden.",
                "tags": ["ml", "ki", "daten", "lernen"]
            },
            {
                "topic": "Backpropagation",
                "question": "Wie funktioniert Backpropagation?",
                "answer": "Backpropagation berechnet die Fehler-Gradienten von der Ausgabeschicht zurück zu den Eingabeschichten, um die Modellgewichte per Gradientenabstieg anzupassen.",
                "tags": ["backpropagation", "gradient", "training", "weights"]
            },
            {
                "topic": "Python Mini AI",
                "question": "Was ist Python Mini AI?",
                "answer": "Python Mini AI ist eine modulare, lokale KI-Anwendung mit eigenem neuronalen Netz, NLP-Verarbeitung, Gedächtnis und Lernsystem.",
                "tags": ["python mini ai", "projekt", "lokal", "chatbot"]
            }
        ]

    def load_knowledge(self):
        """Loads entries from JSON file or creates default knowledge file."""
        if not self.knowledge_file.exists():
            self.entries = self.get_default_knowledge()
            self.save_knowledge()
        else:
            try:
                with open(self.knowledge_file, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
                self.entries = self.get_default_knowledge()

    def save_knowledge(self):
        """Saves entries back to JSON file."""
        self.knowledge_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.knowledge_file, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def add_entry(self, topic: str, question: str, answer: str, tags: List[str] = None) -> bool:
        """Adds a new QA entry to the knowledge base."""
        entry = {
            "topic": topic,
            "question": question,
            "answer": answer,
            "tags": tags or []
        }
        self.entries.append(entry)
        self.save_knowledge()
        return True

    def search(self, query: str) -> Optional[Dict[str, Any]]:
        """Searches for the best matching knowledge entry for a user query."""
        if not self.entries or not query:
            return None

        best_score = 0.0
        best_entry = None

        for entry in self.entries:
            score = self.similarity_calculator.hybrid_similarity(query, entry["question"])

            # Also check if query matches topic or tags
            query_lower = query.lower()
            if entry.get("topic", "").lower() in query_lower:
                score = max(score, 0.75)
            for tag in entry.get("tags", []):
                if tag.lower() in query_lower:
                    score = max(score, 0.65)

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score > 0.25:
            return {
                "entry": best_entry,
                "score": best_score
            }

        return None
