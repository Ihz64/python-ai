import numpy as np
import logging
from typing import Tuple, List, Dict
from python_mini_ai.ai.neural_network import NeuralNetwork
from python_mini_ai.training.dataset import DatasetManager
from python_mini_ai.nlp.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles dataset vectorization and neural network training loop."""

    def __init__(self, dataset_manager: DatasetManager, hidden_dims: List[int] = [64, 32], lr: float = 0.01):
        self.dataset_manager = dataset_manager
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.text_processor = TextProcessor()

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts text patterns and intent tags into numerical X (BoW) and y (one-hot) arrays."""
        vocab = self.dataset_manager.vocabulary
        classes = self.dataset_manager.classes

        X_list = []
        y_list = []

        for intent in self.dataset_manager.intents:
            tag = intent["tag"]
            class_idx = classes.index(tag)

            for pattern in intent["patterns"]:
                tokens = self.text_processor.process(pattern)
                bow = self.text_processor.bag_of_words(tokens, vocab)

                one_hot = [0.0] * len(classes)
                one_hot[class_idx] = 1.0

                X_list.append(bow)
                y_list.append(one_hot)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def train(self, epochs: int = 300, batch_size: int = 8) -> Tuple[NeuralNetwork, List[float]]:
        """Trains the NeuralNetwork model and returns model along with loss history."""
        X, y = self.prepare_training_data()

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Training dataset is empty or vocabulary is empty.")

        input_dim = X.shape[1]
        output_dim = y.shape[1]

        layer_sizes = [input_dim] + self.hidden_dims + [output_dim]
        model = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=self.lr)

        num_samples = X.shape[0]
        loss_history = []

        for epoch in range(epochs):
            # Shuffle dataset
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = int(np.ceil(num_samples / batch_size))

            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_loss = model.train_step(X_batch, y_batch)
                epoch_loss += batch_loss * (end_idx - start_idx)

            epoch_loss /= num_samples
            loss_history.append(epoch_loss)

            if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

        return model, loss_history
