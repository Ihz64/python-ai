import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from python_mini_ai.ai.neural_network import NeuralNetwork
from python_mini_ai.training.dataset import DatasetManager
from python_mini_ai.nlp.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Performs neural network inference on raw text input."""

    def __init__(self, model: NeuralNetwork, dataset_manager: DatasetManager):
        self.model = model
        self.dataset_manager = dataset_manager
        self.text_processor = TextProcessor()

    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Converts text to BoW vector, feeds to neural network, and returns
        predicted intent tag and confidence probability.
        """
        tokens = self.text_processor.process(text)
        vocab = self.dataset_manager.vocabulary
        classes = self.dataset_manager.classes

        if not vocab or not classes:
            return "unknown", 0.0

        bow = self.text_processor.bag_of_words(tokens, vocab)
        X = np.array([bow], dtype=np.float32)

        probs = self.model.predict(X)[0]
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])
        predicted_tag = classes[max_idx]

        return predicted_tag, confidence

    def get_all_intent_probabilities(self, text: str) -> Dict[str, float]:
        """Returns predicted probability for all classes."""
        tokens = self.text_processor.process(text)
        vocab = self.dataset_manager.vocabulary
        classes = self.dataset_manager.classes

        if not vocab or not classes:
            return {}

        bow = self.text_processor.bag_of_words(tokens, vocab)
        X = np.array([bow], dtype=np.float32)

        probs = self.model.predict(X)[0]
        return {cls: float(prob) for cls, prob in zip(classes, probs)}
