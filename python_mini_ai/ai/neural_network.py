import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

class NeuralNetwork:
    """
    Custom Feedforward Neural Network (Multi-Layer Perceptron) built with pure NumPy.
    Supports ReLU hidden activations, Softmax output, Categorical Cross-Entropy, Adam/SGD optimizer,
    Forward & Backpropagation, and weight saving/loading.
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        layer_sizes: e.g. [input_dim, hidden1, hidden2, output_dim]
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Adam optimizer parameters
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
        """He initialization for ReLU layers."""
        self.weights = []
        self.biases = []
        self.m_w, self.v_w, self.m_b, self.v_b = [], [], [], []

        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]
            # He normal initialization
            w = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
            b = np.zeros((1, out_dim))
            self.weights.append(w)
            self.biases.append(b)

            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation through the network.
        Returns:
            activations: List of activations for each layer (including input).
            z_values: List of pre-activation linear combinations for hidden/output layers.
        """
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
                # Output layer: Softmax
                current_a = self.softmax(z)
            else:
                # Hidden layer: ReLU
                current_a = self.relu(z)

            activations.append(current_a)

        return activations, z_values

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-Entropy loss function."""
        clipped_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(np.sum(y_true * np.log(clipped_pred), axis=1)))

    def train_step(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Performs one forward pass, computes loss, backpropagates gradients, and updates parameters.
        """
        batch_size = X.shape[0]
        activations, z_values = self.forward(X)
        y_pred = activations[-1]
        loss = self.compute_loss(y_pred, y_true)

        # Backpropagation
        # Gradient for output layer (Softmax + Cross-Entropy)
        dz = (y_pred - y_true) / batch_size

        num_layers = len(self.weights)
        dw_list = [None] * num_layers
        db_list = [None] * num_layers

        for l in reversed(range(num_layers)):
            a_prev = activations[l]
            dw = np.dot(a_prev.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            dw_list[l] = dw
            db_list[l] = db

            if l > 0:
                da_prev = np.dot(dz, self.weights[l].T)
                dz = da_prev * self.relu_derivative(z_values[l - 1])

        # Adam Optimizer Update
        self.t += 1
        for i in range(num_layers):
            # Update weights with Adam
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw_list[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dw_list[i] ** 2)
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

            # Update biases with Adam
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db_list[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db_list[i] ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probabilities for input X."""
        activations, _ = self.forward(X)
        return activations[-1]

    def save(self, filepath: Path or str):
        """Saves weights and biases to an .npz file."""
        data_dict = {}
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            data_dict[f"w_{idx}"] = w
            data_dict[f"b_{idx}"] = b
        data_dict["layer_sizes"] = np.array(self.layer_sizes)
        np.savez(filepath, **data_dict)

    def load(self, filepath: Path or str) -> bool:
        """Loads weights and biases from an .npz file if present."""
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
