import math
from typing import List, Dict, Set
from python_mini_ai.nlp.text_processor import TextProcessor

class SimilarityCalculator:
    """Calculates text similarity using Cosine Similarity and Jaccard Index."""

    def __init__(self, text_processor: TextProcessor = None):
        self.processor = text_processor or TextProcessor(remove_stopwords=False)

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculates Jaccard Similarity between two texts."""
        tokens1 = set(self.processor.process(text1))
        tokens2 = set(self.processor.process(text2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)

    def cosine_similarity_bow(self, text1: str, text2: str) -> float:
        """Calculates Cosine Similarity using bag-of-words representation."""
        tokens1 = self.processor.process(text1)
        tokens2 = self.processor.process(text2)

        if not tokens1 or not tokens2:
            return 0.0

        vocab = sorted(list(set(tokens1 + tokens2)))
        vec1 = self.processor.bag_of_words(tokens1, vocab)
        vec2 = self.processor.bag_of_words(tokens2, vocab)

        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm1 = math.sqrt(sum(v1 ** 2 for v1 in vec1))
        norm2 = math.sqrt(sum(v2 ** 2 for v2 in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def hybrid_similarity(self, text1: str, text2: str) -> float:
        """Combines Jaccard and Cosine similarity for better matching."""
        jaccard = self.jaccard_similarity(text1, text2)
        cosine = self.cosine_similarity_bow(text1, text2)
        return 0.4 * jaccard + 0.6 * cosine
