import pytest
import numpy as np
from pathlib import Path
from python_mini_ai_ultra.ai.ensemble_network import UltraEnsembleNeuralNetwork
from python_mini_ai_ultra.knowledge.dataset_generator import generate_ultra_dataset

def test_ultra_ensemble_network():
    net = UltraEnsembleNeuralNetwork(layer_sizes=[10, 20, 5])
    X = np.random.randn(2, 10)
    y = np.eye(2, 5)
    loss = net.train_step(X, y)
    assert loss > 0
    pred = net.predict(X)
    assert pred.shape == (2, 5)

def test_ultra_dataset_generator(tmp_path):
    k_path = tmp_path / "ultra_k.json"
    t_path = tmp_path / "ultra_t.json"
    generate_ultra_dataset(k_path, t_path)
    assert k_path.exists()
    assert t_path.exists()
