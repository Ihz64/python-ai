import re
from typing import List, Set

class Tokenizer:
    """Lightweight custom tokenizer for German and English text with pre-compiled regex."""

    # Pre-compiled regex pattern for high-speed tokenization
    TOKEN_PATTERN = re.compile(r'\b[a-zA-Z0-9äöüÄÖÜß]+\b')

    def __init__(self, lower: bool = True):
        self.lower = lower

    def tokenize(self, text: str) -> List[str]:
        """Splits input text into word tokens using optimized regex matching."""
        if not text:
            return []

        if self.lower:
            text = text.lower()

        return self.TOKEN_PATTERN.findall(text)
