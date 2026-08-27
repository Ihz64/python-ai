import random
from typing import List, Dict, Optional
from python_mini_ai.training.dataset import DatasetManager

class ResponseGenerator:
    """Generates responses based on intent tags or fallback templates."""

    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager

    def generate_response(self, intent_tag: str) -> str:
        """Retrieves a random response configured for the given intent tag."""
        for intent in self.dataset_manager.intents:
            if intent["tag"] == intent_tag:
                responses = intent.get("responses", [])
                if responses:
                    return random.choice(responses)

        return self.fallback_response()

    def fallback_response(self) -> str:
        """Fallback response when no specific intent or response is found."""
        fallbacks = [
            "Das ist ein interessanter Gedanke, erzähl mir gerne mehr darüber.",
            "Ich habe deine Eingabe verstanden, bin mir aber nicht ganz sicher, was du genau wissen möchtest.",
            "Kannst du das bitte noch etwas genauer formulieren?",
            "Ich lerne noch ständig dazu. Was genau bedeutet das für dich?"
        ]
        return random.choice(fallbacks)
