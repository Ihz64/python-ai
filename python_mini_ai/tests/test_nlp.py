import pytest
from python_mini_ai.nlp.tokenizer import Tokenizer
from python_mini_ai.nlp.text_processor import TextProcessor
from python_mini_ai.nlp.similarity import SimilarityCalculator

def test_tokenizer():
    tokenizer = Tokenizer(lower=True)
    tokens = tokenizer.tokenize("Hallo Welt! Wie geht es dir?")
    assert tokens == ["hallo", "welt", "wie", "geht", "es", "dir"]

def test_text_processor():
    processor = TextProcessor(remove_stopwords=True)
    tokens = processor.process("Das ist ein Test für Python.")
    assert "test" in tokens
    assert "python" in tokens
    assert "ist" not in tokens  # Stop word removed

def test_similarity_calculator():
    sim = SimilarityCalculator()
    score1 = sim.hybrid_similarity("Was ist Python?", "Erkläre Python")
    score2 = sim.hybrid_similarity("Was ist Python?", "Rezept für Kuchen")
    assert score1 > score2
    assert score1 > 0.2
