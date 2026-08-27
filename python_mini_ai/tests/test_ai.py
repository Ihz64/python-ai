import numpy as np
import pytest
from pathlib import Path
from python_mini_ai.ai.neural_network import NeuralNetwork
from python_mini_ai.training.dataset import DatasetManager
from python_mini_ai.ai.trainer import ModelTrainer
from python_mini_ai.ai.inference import InferenceEngine
from python_mini_ai.ai.intent_classifier import IntentClassifier
from python_mini_ai.ai.decision_engine import DecisionEngine
from python_mini_ai.ai.response_generator import ResponseGenerator

def test_neural_network_forward_pass():
    nn = NeuralNetwork(layer_sizes=[4, 8, 3])
    X = np.random.randn(2, 4)
    activations, z_values = nn.forward(X)
    assert len(activations) == 3
    assert activations[-1].shape == (2, 3)
    assert np.allclose(np.sum(activations[-1], axis=1), 1.0)

def test_neural_network_training_step():
    nn = NeuralNetwork(layer_sizes=[4, 8, 2], learning_rate=0.05)
    X = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    y = np.array([[1.0, 0.0]], dtype=np.float32)

    loss1 = nn.train_step(X, y)
    for _ in range(50):
        loss2 = nn.train_step(X, y)

    assert loss2 < loss1

def test_decision_engine_pipeline(tmp_path):
    dataset_file = tmp_path / "training_data.json"
    dm = DatasetManager(dataset_file)
    trainer = ModelTrainer(dm, hidden_dims=[16, 8], lr=0.01)
    model, _ = trainer.train(epochs=20)

    infer = InferenceEngine(model, dm)
    clf = IntentClassifier(infer)
    rg = ResponseGenerator(dm)
    engine = DecisionEngine(clf, rg)

    result = engine.process_query("Hallo!")
    assert "response" in result
    assert result["confidence"] > 0.0
    assert result["source"] in ["neural_network", "knowledge_base", "fallback"]
