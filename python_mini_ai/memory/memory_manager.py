from pathlib import Path
from typing import List, Dict, Any, Optional
from python_mini_ai.memory.short_term import ShortTermMemory
from python_mini_ai.memory.long_term import LongTermMemory

class MemoryManager:
    """Facade for managing both Short-Term (in-memory) and Long-Term (SQLite) memory systems."""

    def __init__(self, db_path: Path, max_short_term: int = 10):
        self.short_term = ShortTermMemory(max_size=max_short_term)
        self.long_term = LongTermMemory(db_path=db_path)

    def add_short_term(self, role: str, content: str, extra: Dict[str, Any] = None):
        """Adds message turn to short-term memory and logs to long-term database."""
        self.short_term.add_message(role, content, extra)
        confidence = extra.get("confidence", 1.0) if extra else 1.0
        self.long_term.save_chat_turn(role, content, confidence)

    def get_context(self) -> List[Dict[str, Any]]:
        """Returns recent short-term context list."""
        return self.short_term.get_context_list()

    def search_long_term_qa(self, query: str) -> Optional[Dict[str, Any]]:
        """Searches learned QA in SQLite long-term storage."""
        return self.long_term.search_learned_qa(query)

    def add_learned_fact(self, question: str, answer: str, score: float = 1.0):
        """Saves a learned Q&A fact to long term memory."""
        self.long_term.add_learned_qa(question, answer, score)

    def clear_short_term(self):
        """Clears current active short term conversation."""
        self.short_term.clear()
