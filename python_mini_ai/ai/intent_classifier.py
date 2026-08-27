from typing import Tuple, Dict, Any
from python_mini_ai.ai.inference import InferenceEngine

class IntentClassifier:
    """Classifies user intent using the trained neural network inference engine."""

    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies input text and returns structured classification results.
        """
        intent, confidence = self.inference_engine.predict_intent(text)
        all_probs = self.inference_engine.get_all_intent_probabilities(text)

        return {
            "intent": intent,
            "confidence": confidence,
            "all_probabilities": all_probs
        }
