import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

class UltraEnsembleNeuralNetwork:
    """
    Multi-Architecture Neural Ensemble Network.
    Combines a primary Deep MLP with an Attention-weighted Query Layer
    for enhanced local pattern classification and confidence estimation.
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.005):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        self.mlp_weights: List[np.ndarray] = []
        self.mlp_biases: List[np.ndarray] = []
        self.attention_w = np.random.randn(layer_sizes[0], layer_sizes[0]) * 0.1

        self._init_ensemble()

    def _init_ensemble(self):
        self.mlp_weights = []
        self.mlp_biases = []
        for i in range(len(self.layer_sizes) - 1):
            in_d = self.layer_sizes[i]
            out_d = self.layer_sizes[i + 1]
            w = np.random.randn(in_d, out_d) * np.sqrt(2.0 / in_d)
            b = np.zeros((1, out_d))
            self.mlp_weights.append(w)
            self.mlp_biases.append(b)
        self.attention_w = np.random.randn(self.layer_sizes[0], self.layer_sizes[0]) * 0.1

    def apply_attention(self, X: np.ndarray) -> np.ndarray:
        """Applies self-attention query weights to input features."""
        if X.shape[1] != self.attention_w.shape[0]:
            self.layer_sizes[0] = X.shape[1]
            self.attention_w = np.random.randn(X.shape[1], X.shape[1]) * 0.1
            self._init_ensemble()

        attn_scores = np.dot(X, self.attention_w)
        exp_scores = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return X * attn_weights

    def forward_full(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        attn_x = self.apply_attention(X)
        activations = [attn_x]
        z_values = []
        curr = attn_x
        num_layers = len(self.mlp_weights)

        for i in range(num_layers):
            w = self.mlp_weights[i]
            b = self.mlp_biases[i]
            z = np.dot(curr, w) + b
            z_values.append(z)

            if i == num_layers - 1:
                exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
                curr = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
            else:
                curr = np.where(z > 0, z, 0.01 * z)

            activations.append(curr)

        return activations, z_values

    def forward(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward_full(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        batch_size = X.shape[0]
        activations, z_values = self.forward_full(X)
        y_pred = activations[-1]
        clipped = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        loss = float(-np.mean(np.sum(y * np.log(clipped), axis=1)))

        dz = (y_pred - y) / batch_size
        num_layers = len(self.mlp_weights)

        for l in reversed(range(num_layers)):
            a_prev = activations[l]
            dw = np.dot(a_prev.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            self.mlp_weights[l] -= self.learning_rate * np.clip(dw, -2.0, 2.0)
            self.mlp_biases[l] -= self.learning_rate * db

            if l > 0:
                da_prev = np.dot(dz, self.mlp_weights[l].T)
                z_prev = z_values[l - 1]
                dz = np.where(z_prev > 0, da_prev, 0.01 * da_prev)

        # Backpropagation to attention matrix
        if num_layers > 0:
            da_attn = np.dot(dz, self.mlp_weights[0].T)
            d_attn_w = np.dot(X.T, da_attn)
            if d_attn_w.shape == self.attention_w.shape:
                self.attention_w -= self.learning_rate * np.clip(d_attn_w, -1.0, 1.0)

        return loss

    def save(self, filepath: Path or str):
        data = {}
        for idx, (w, b) in enumerate(zip(self.mlp_weights, self.mlp_biases)):
            data[f"w_{idx}"] = w
            data[f"b_{idx}"] = b
        data["layer_sizes"] = np.array(self.layer_sizes)
        data["attention_w"] = self.attention_w
        np.savez(filepath, **data)

    def load(self, filepath: Path or str) -> bool:
        path = Path(filepath)
        if not path.exists():
            return False
        data = np.load(path)
        self.layer_sizes = list(data["layer_sizes"])
        self.attention_w = data["attention_w"]
        self.mlp_weights, self.mlp_biases = [], []
        idx = 0
        while f"w_{idx}" in data:
            self.mlp_weights.append(data[f"w_{idx}"])
            self.mlp_biases.append(data[f"b_{idx}"])
            idx += 1
        return True
