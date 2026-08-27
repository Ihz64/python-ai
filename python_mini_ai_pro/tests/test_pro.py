import pytest
import numpy as np
from python_mini_ai_pro.ai.neural_network import ProNeuralNetwork
from python_mini_ai_pro.rag.vector_store import VectorStoreRAG
from python_mini_ai_pro.sandbox.code_sandbox import CodeSandbox

def test_pro_neural_network():
    net = ProNeuralNetwork(layer_sizes=[8, 16, 4], learning_rate=0.01)
    X = np.random.randn(4, 8)
    y = np.eye(4, 4)
    loss1 = net.train_step(X, y)
    for _ in range(20):
        loss2 = net.train_step(X, y)
    assert loss2 < loss1

def test_pro_rag_vector_store(tmp_path):
    store_file = tmp_path / "vector_store.json"
    rag = VectorStoreRAG(store_file)
    rag.add_document("doc1", "Quantik", "Quantenmechanik beschreibt atomare Physik.")

    results = rag.search("Quantenmechanik")
    assert len(results) > 0
    assert results[0]["doc_id"] == "doc1"

def test_pro_code_sandbox():
    sandbox = CodeSandbox()
    res = sandbox.execute_python_code("res = 10 * 10\nprint('Ergebnis:', res)")
    assert res["success"] is True
    assert "Ergebnis: 100" in res["output"]
    assert res["local_variables"]["res"] == "100"
