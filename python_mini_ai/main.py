import sys
import os
import logging
from pathlib import Path

from python_mini_ai import config
from python_mini_ai.training.dataset import DatasetManager
from python_mini_ai.knowledge.knowledge_base import KnowledgeBase
from python_mini_ai.memory.memory_manager import MemoryManager
from python_mini_ai.ai.neural_network import NeuralNetwork
from python_mini_ai.ai.trainer import ModelTrainer
from python_mini_ai.ai.inference import InferenceEngine
from python_mini_ai.ai.intent_classifier import IntentClassifier
from python_mini_ai.ai.response_generator import ResponseGenerator
from python_mini_ai.ai.decision_engine import DecisionEngine
from python_mini_ai.training.learning_system import LearningSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PythonMiniAI")

def run_cli_mode():
    """Fallback interactive CLI chat loop for terminal/headless environments."""
    print("=" * 60)
    print("   Python Mini AI - Local Intelligence Terminal Chatbot   ")
    print("=" * 60)
    print("Initialisiere lokale KI-Komponenten...\n")

    dataset_manager = DatasetManager(config.TRAINING_DATA_PATH)
    knowledge_base = KnowledgeBase(config.KNOWLEDGE_PATH)
    memory_manager = MemoryManager(config.DB_PATH, max_short_term=config.SHORT_TERM_MEMORY_SIZE)

    model = NeuralNetwork(layer_sizes=[10, 64, 32, 5])
    if config.MODEL_PATH.exists():
        model.load(config.MODEL_PATH)
    else:
        print("Trainiere neuronales Netzwerk erstmaig...")
        trainer = ModelTrainer(dataset_manager, hidden_dims=config.HIDDEN_DIMS, lr=config.LEARNING_RATE)
        model, _ = trainer.train(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
        model.save(config.MODEL_PATH)
        print("Modell erfolgreich gespeichert!\n")

    inference_engine = InferenceEngine(model, dataset_manager)
    intent_classifier = IntentClassifier(inference_engine)
    response_generator = ResponseGenerator(dataset_manager)
    decision_engine = DecisionEngine(
        intent_classifier=intent_classifier,
        response_generator=response_generator,
        knowledge_base=knowledge_base,
        memory_manager=memory_manager
    )
    learning_system = LearningSystem(
        dataset_manager=dataset_manager,
        memory_manager=memory_manager,
        knowledge_base=knowledge_base,
        model=model
    )

    print("KI Bereitschaft hergestellt. Gib 'exit' oder 'quit' ein zum Beenden.\n")

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "beenden"]:
                print("Auf Wiedersehen!")
                break

            result = decision_engine.process_query(user_input)
            response = result["response"]
            confidence = result["confidence"]
            source = result["source"]

            print(f"AI   > {response}")
            print(f"       [Confidence: {confidence * 100:.1f}% | Source: {source}]")

        except (KeyboardInterrupt, EOFError):
            print("\nBeende Program...")
            break

def main():
    """Main application entry point."""
    # Check if GUI display is available
    display = os.environ.get("DISPLAY")
    if display or sys.platform in ["win32", "darwin"]:
        try:
            from python_mini_ai.ui.chat_window import launch_gui
            launch_gui()
        except Exception as e:
            logger.warning(f"GUI konnte nicht gestartet werden ({e}). Wechsle in den CLI-Modus.")
            run_cli_mode()
    else:
        logger.info("Kein Display gefunden. Starte CLI-Modus.")
        run_cli_mode()

if __name__ == "__main__":
    main()
