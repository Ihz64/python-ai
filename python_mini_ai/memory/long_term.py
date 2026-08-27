import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from python_mini_ai.nlp.similarity import SimilarityCalculator

logger = logging.getLogger(__name__)

class LongTermMemory:
    """Persistent SQLite database for storing user preferences, learned QA, and chat history."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.similarity_calculator = SimilarityCalculator()
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Creates required SQLite tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Chat History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # User Preferences / Facts Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Learned Q&A Pairs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_qa (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    score REAL DEFAULT 1.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def save_chat_turn(self, role: str, content: str, confidence: float = 1.0):
        """Persists a chat message turn."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (role, content, confidence) VALUES (?, ?, ?)",
                (role, content, confidence)
            )
            conn.commit()

    def set_preference(self, key: str, value: str):
        """Stores or updates a user preference key-value pair."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_preferences (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
                (key, value)
            )
            conn.commit()

    def get_preference(self, key: str) -> Optional[str]:
        """Gets a user preference value by key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM user_preferences WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def add_learned_qa(self, question: str, answer: str, score: float = 1.0):
        """Stores a user-provided or feedback-confirmed Q&A pair."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO learned_qa (question, answer, score) VALUES (?, ?, ?)",
                (question, answer, score)
            )
            conn.commit()

    def search_learned_qa(self, query: str) -> Optional[Dict[str, Any]]:
        """Searches learned Q&A pairs using similarity calculator."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT question, answer, score FROM learned_qa")
            rows = cursor.fetchall()

        if not rows:
            return None

        best_match = None
        best_similarity = 0.0

        for question, answer, score in rows:
            sim = self.similarity_calculator.hybrid_similarity(query, question)
            if sim > best_similarity:
                best_similarity = sim
                best_match = {"question": question, "answer": answer, "score": sim}

        if best_match and best_similarity > 0.30:
            return best_match

        return None

    def get_recent_chat_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches recent stored chat messages."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, confidence, timestamp FROM chat_history "
                "ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = cursor.fetchall()

        history = [
            {"role": r[0], "content": r[1], "confidence": r[2], "timestamp": r[3]}
            for r in reversed(rows)
        ]
        return history
