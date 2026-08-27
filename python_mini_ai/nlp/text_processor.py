import re
from typing import List, Set
from python_mini_ai.nlp.tokenizer import Tokenizer

# Common German and English stop words
STOP_WORDS: Set[str] = {
    "der", "die", "das", "ein", "eine", "einer", "einem", "einen", "und", "oder", "aber",
    "ist", "sind", "war", "waren", "ich", "du", "er", "sie", "es", "wir", "ihr", "sie",
    "in", "im", "auf", "mit", "von", "für", "zu", "zum", "zur", "an", "am",
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "i", "you", "he", "she", "it",
    "we", "they", "in", "on", "at", "for", "to", "with", "by", "of"
}

class TextProcessor:
    """Normalizes text and builds bag-of-words vectors."""

    def __init__(self, remove_stopwords: bool = False):
        self.tokenizer = Tokenizer(lower=True)
        self.remove_stopwords = remove_stopwords

    def normalize(self, text: str) -> str:
        """Normalizes text by removing extra spaces and special characters."""
        if not text:
            return ""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def process(self, text: str) -> List[str]:
        """Normalizes and tokenizes text, optionally removing stopwords."""
        normalized = self.normalize(text)
        tokens = self.tokenizer.tokenize(normalized)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOP_WORDS]
        return tokens

    def bag_of_words(self, tokens: List[str], vocabulary: List[str]) -> List[float]:
        """Generates a binary bag-of-words vector for given tokens against a vocabulary."""
        token_set = set(tokens)
        return [1.0 if word in token_set else 0.0 for word in vocabulary]
