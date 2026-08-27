import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

class ProNeuralNetwork:
    """
    Advanced Deep Feedforward Neural Network in pure NumPy for Python Mini AI Pro.
    Features:
    - Multi-layer architecture with arbitrary hidden layer dimensions
    - LeakyReLU and Softmax activations
    - He (Kaiming) normal initialization
    - Adam Optimizer with Bias Correction and L2 Weight Decay
    - Gradient Clipping to prevent exploding gradients
    - Model Weight saving, loading, and parameter summary
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.005, weight_decay: float = 1e-4):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Adam optimizer state arrays
        self.m_w: List[np.ndarray] = []
        self.v_w: List[np.ndarray] = []
        self.m_b: List[np.ndarray] = []
        self.v_b: List[np.ndarray] = []
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        self._initialize_weights()

    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        self.m_w, self.v_w, self.m_b, self.v_b = [], [], [], []

        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]
            w = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
            b = np.zeros((1, out_dim))

            self.weights.append(w)
            self.biases.append(b)

            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        dx = np.ones_like(x)
        dx[x <= 0] = alpha
        return dx

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        z_values = []
        current_a = X
        num_layers = len(self.weights)

        for i in range(num_layers):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(current_a, w) + b
            z_values.append(z)

            if i == num_layers - 1:
                current_a = self.softmax(z)
            else:
                current_a = self.leaky_relu(z)

            activations.append(current_a)

        return activations, z_values

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        clipped_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        cross_entropy = -np.mean(np.sum(y_true * np.log(clipped_pred), axis=1))

        # Add L2 Regularization Loss
        l2_loss = 0.5 * self.weight_decay * sum(np.sum(w ** 2) for w in self.weights)
        return float(cross_entropy + l2_loss)

    def train_step(self, X: np.ndarray, y_true: np.ndarray, max_grad_norm: float = 5.0) -> float:
        batch_size = X.shape[0]
        activations, z_values = self.forward(X)
        y_pred = activations[-1]
        loss = self.compute_loss(y_pred, y_true)

        dz = (y_pred - y_true) / batch_size
        num_layers = len(self.weights)
        dw_list = [None] * num_layers
        db_list = [None] * num_layers

        for l in reversed(range(num_layers)):
            a_prev = activations[l]
            dw = np.dot(a_prev.T, dz) + self.weight_decay * self.weights[l]
            db = np.sum(dz, axis=0, keepdims=True)

            # Gradient Clipping
            dw_norm = np.linalg.norm(dw)
            if dw_norm > max_grad_norm:
                dw = dw * (max_grad_norm / dw_norm)

            dw_list[l] = dw
            db_list[l] = db

            if l > 0:
                da_prev = np.dot(dz, self.weights[l].T)
                dz = da_prev * self.leaky_relu_derivative(z_values[l - 1])

        self.t += 1
        for i in range(num_layers):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw_list[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dw_list[i] ** 2)
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db_list[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db_list[i] ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return activations[-1]

    def save(self, filepath: Path or str):
        data_dict = {}
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            data_dict[f"w_{idx}"] = w
            data_dict[f"b_{idx}"] = b
        data_dict["layer_sizes"] = np.array(self.layer_sizes)
        np.savez(filepath, **data_dict)

    def load(self, filepath: Path or str) -> bool:
        path = Path(filepath)
        if not path.exists():
            return False

        data = np.load(path)
        self.layer_sizes = list(data["layer_sizes"])
        self.weights = []
        self.biases = []
        self.m_w, self.v_w, self.m_b, self.v_b = [], [], [], []

        idx = 0
        while f"w_{idx}" in data:
            w = data[f"w_{idx}"]
            b = data[f"b_{idx}"]
            self.weights.append(w)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))
            idx += 1

        self.t = 0
        return True
