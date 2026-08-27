import re
from typing import List

class Tokenizer:
    """Lightweight custom tokenizer for German and English text."""

    def __init__(self, lower: bool = True):
        self.lower = lower

    def tokenize(self, text: str) -> List[str]:
        """Splits input text into word tokens, removing punctuation."""
        if not text:
            return []

        if self.lower:
            text = text.lower()

        # Keep alphanumeric characters and German umlauts (ä, ö, ü, ß)
        tokens = re.findall(r'\b[a-zA-Z0-9äöüÄÖÜß]+\b', text)
        return tokens
