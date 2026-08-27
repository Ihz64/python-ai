import pytest
from pathlib import Path
from python_mini_ai.knowledge.knowledge_base import KnowledgeBase
from python_mini_ai.memory.memory_manager import MemoryManager

def test_knowledge_base(tmp_path):
    kb_file = tmp_path / "knowledge.json"
    kb = KnowledgeBase(kb_file)
    kb.add_entry(topic="Pytest", question="Was ist Pytest Framework?", answer="Ein bekanntes Test Framework.", tags=["pytest"])

    result = kb.search("Was ist Pytest Framework?")
    assert result is not None
    assert result["entry"]["answer"] == "Ein bekanntes Test Framework."
    assert result["score"] >= 0.8

def test_memory_manager(tmp_path):
    db_file = tmp_path / "memory.db"
    mm = MemoryManager(db_file, max_short_term=3)

    mm.add_short_term("user", "Hallo")
    mm.add_short_term("assistant", "Hi!")
    context = mm.get_context()
    assert len(context) == 2

    mm.add_learned_fact("Was ist Pytest?", "Ein Test Framework.")
    learned = mm.search_long_term_qa("Was ist Pytest?")
    assert learned is not None
    assert learned["answer"] == "Ein Test Framework."
