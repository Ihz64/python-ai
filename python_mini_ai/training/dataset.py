import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from python_mini_ai.nlp.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages intent training patterns, vocabulary, and category labels."""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.text_processor = TextProcessor()
        self.vocabulary: List[str] = []
        self.classes: List[str] = []
        self.intents: List[Dict] = []
        self.load_data()

    def get_default_dataset(self) -> Dict:
        return {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hallo", "Hi", "Guten Tag", "Moin", "Hey", "Servus", "Hello", "Good morning"],
                    "responses": ["Hallo! Wie kann ich dir heute helfen?", "Guten Tag! Was kann ich für dich tun?", "Hey! Schon von dir zu hören."]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["Tschüss", "Auf Wiedersehen", "Bis später", "Ciao", "Bye", "Bis bald"],
                    "responses": ["Auf Wiedersehen! Einen schönen Tag noch.", "Tschüss! Melde dich gerne jederzeit wieder.", "Bis bald!"]
                },
                {
                    "tag": "thanks",
                    "patterns": ["Danke", "Vielen Dank", "Dankeschön", "Thanks", "Super danke"],
                    "responses": ["Sehr gerne!", "Nichts zu danken! Immer wieder gerne.", "Freut mich, dass ich helfen konnte!"]
                },
                {
                    "tag": "python_info",
                    "patterns": ["Was ist Python", "Erkläre Python", "Python Programmierung", "Wie funktioniert Python"],
                    "responses": ["Python ist eine vielseitige, leicht zu erlernende Programmiersprache, die in Webentwicklung, Data Science und KI verwendet wird."]
                },
                {
                    "tag": "help",
                    "patterns": ["Hilfe", "Was kannst du tun", "Was sind deine Funktionen", "Wie hilfst du mir"],
                    "responses": ["Ich bin die Python Mini AI. Ich kann dir Fragen beantworten, über Python sprechen und aus deinen Eingaben lernen."]
                },
                {
                    "tag": "weather",
                    "patterns": ["Wie ist das Wetter", "Regnet es heute", "Wetterbericht", "Wetter heute"],
                    "responses": ["Als lokale KI habe ich keinen direkten Wetterdienst, aber ich hoffe, bei dir scheint die Sonne!"]
                },
                {
                    "tag": "math",
                    "patterns": ["Kannst du rechnen", "Mathematik", "Mathe Hilfe"],
                    "responses": ["Ja, einfache mathematische Logik und Fragen kann ich mit dir durchgehen!"]
                },
                {
                    "tag": "unknown",
                    "patterns": ["xyz123", "quwertz", "unbekanntes muster"],
                    "responses": ["Das habe ich leider nicht ganz verstanden. Kannst du es genauer erklären?"]
                }
            ]
        }

    def load_data(self):
        """Loads dataset from JSON file or creates default dataset."""
        if not self.data_path.exists():
            data = self.get_default_dataset()
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        self.intents = data.get("intents", [])
        self._build_vocab_and_classes()

    def save_data(self):
        """Saves current intents back to JSON file."""
        data = {"intents": self.intents}
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _build_vocab_and_classes(self):
        """Extracts unique words and class labels."""
        vocab_set = set()
        class_set = set()

        for intent in self.intents:
            tag = intent["tag"]
            class_set.add(tag)
            for pattern in intent["patterns"]:
                tokens = self.text_processor.process(pattern)
                vocab_set.update(tokens)

        self.vocabulary = sorted(list(vocab_set))
        self.classes = sorted(list(class_set))

    def add_example(self, tag: str, pattern: str, response: Optional[str] = None) -> bool:
        """Adds a new training example to an existing or new intent tag."""
        intent = next((i for i in self.intents if i["tag"] == tag), None)
        if intent:
            if pattern not in intent["patterns"]:
                intent["patterns"].append(pattern)
            if response and response not in intent["responses"]:
                intent["responses"].append(response)
        else:
            self.intents.append({
                "tag": tag,
                "patterns": [pattern],
                "responses": [response] if response else ["Ich habe das gelernt!"]
            })

        self.save_data()
        self._build_vocab_and_classes()
        return True
