import pytest
from python_mini_ai.ai.math_engine import MathEngine

def test_math_engine_basic():
    me = MathEngine()
    res1 = me.extract_and_evaluate("Was ist 12 + 28?")
    assert res1 is not None
    assert res1["result"] == 40
    assert res1["confidence"] == 1.0

def test_math_engine_multiplication():
    me = MathEngine()
    res2 = me.extract_and_evaluate("Berechne 15 * 6")
    assert res2 is not None
    assert res2["result"] == 90

def test_math_engine_exponent_and_sqrt():
    me = MathEngine()
    res3 = me.extract_and_evaluate("sqrt(144)")
    assert res3 is not None
    assert res3["result"] == 12

def test_math_engine_non_math():
    me = MathEngine()
    res4 = me.extract_and_evaluate("Wie heiße ich?")
    assert res4 is None
