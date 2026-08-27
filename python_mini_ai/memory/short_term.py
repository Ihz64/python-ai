from collections import deque
from typing import List, Dict, Any

class ShortTermMemory:
    """In-memory sliding window queue for ongoing chat turn history."""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.history: deque = deque(maxlen=max_size)

    def add_message(self, role: str, content: str, extra: Dict[str, Any] = None):
        """Appends a new turn to short-term context."""
        entry = {
            "role": role,
            "content": content,
            "extra": extra or {}
        }
        self.history.append(entry)

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns list of recent messages."""
        return list(self.history)

    def get_context_list(self) -> List[Dict[str, Any]]:
        """Returns list of recent messages."""
        return list(self.history)

    def clear(self):
        """Clears short-term memory."""
        self.history.clear()
