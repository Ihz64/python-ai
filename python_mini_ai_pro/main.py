import sys
import os
import logging
from pathlib import Path

from python_mini_ai_pro import config
from python_mini_ai_pro.ai.neural_network import ProNeuralNetwork
from python_mini_ai.ai.math_engine import MathEngine
from python_mini_ai_pro.rag.vector_store import VectorStoreRAG
from python_mini_ai_pro.sandbox.code_sandbox import CodeSandbox
from python_mini_ai.training.dataset import DatasetManager
from python_mini_ai.knowledge.knowledge_base import KnowledgeBase
from python_mini_ai.memory.memory_manager import MemoryManager
from python_mini_ai.ai.trainer import ModelTrainer
from python_mini_ai.ai.inference import InferenceEngine
from python_mini_ai.ai.intent_classifier import IntentClassifier
from python_mini_ai.ai.response_generator import ResponseGenerator
from python_mini_ai.ai.decision_engine import DecisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PythonMiniAIPro")

def run_pro_cli():
    """Terminal chat interface for Python Mini AI Pro."""
    print("=" * 70)
    print("   🚀 PYTHON MINI AI PRO - LOCAL DEEP INTELLIGENCE SUITE v2.0   ")
    print("=" * 70)
    print("Initialisiere Pro-Komponenten (Deep Net, RAG VectorStore, Code Sandbox)...\n")

    dataset_manager = DatasetManager(config.TRAINING_DATA_PATH)
    knowledge_base = KnowledgeBase(config.KNOWLEDGE_PATH)
    memory_manager = MemoryManager(config.DB_PATH, max_short_term=config.SHORT_TERM_MEMORY_SIZE)
    vector_store = VectorStoreRAG(config.VECTOR_STORE_PATH)
    sandbox = CodeSandbox()

    vector_store.add_document("doc_phys", "Physik & Quanten", "Die Quantenphysik beschreibt das Verhalten von Subatomaren Teilchen.")
    vector_store.add_document("doc_py", "Python Pro Guide", "Python Pro unterstützt RAG Vektorspeicher, Code Ausführung und neuronale Netze.")

    pro_net = ProNeuralNetwork(layer_sizes=[10, 64, 32, 5])
    if config.MODEL_PATH.exists():
        pro_net.load(config.MODEL_PATH)
    else:
        print("Trainiere Pro Deep Neural Network...")
        trainer = ModelTrainer(dataset_manager, hidden_dims=config.HIDDEN_DIMS, lr=config.LEARNING_RATE)
        base_model, _ = trainer.train(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
        pro_net.weights = base_model.weights
        pro_net.biases = base_model.biases
        pro_net.save(config.MODEL_PATH)
        print("Pro Modell erfolgreich trainiert & gespeichert!")

    inference_engine = InferenceEngine(pro_net, dataset_manager)
    intent_classifier = IntentClassifier(inference_engine)
    response_generator = ResponseGenerator(dataset_manager)
    decision_engine = DecisionEngine(
        intent_classifier=intent_classifier,
        response_generator=response_generator,
        knowledge_base=knowledge_base,
        memory_manager=memory_manager,
        math_engine=MathEngine()
    )

    print("\nPRO KI Bereit! Gib 'run <code>' für Sandbox oder 'search <query>' für RAG ein.")
    print("Gib 'exit' ein zum Beenden.\n")

    while True:
        try:
            user_input = input("\nPRO-AI User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "beenden"]:
                print("Auf Wiedersehen von Python Mini AI Pro!")
                break

            if user_input.lower().startswith("run "):
                code_snippet = user_input[4:]
                res = sandbox.execute_python_code(code_snippet)
                print(f"\n[SANDBOX EXECUTION]")
                print(f"Status: {'SUCCESS' if res['success'] else 'ERROR'}")
                print(f"Output:\n{res['output']}")
                if res['error']:
                    print(f"Error:\n{res['error']}")
                continue

            if user_input.lower().startswith("search "):
                query = user_input[7:]
                rag_matches = vector_store.search(query)
                print(f"\n[RAG VECTOR RETRIEVAL]")
                for idx, m in enumerate(rag_matches, 1):
                    print(f"{idx}. [{m['title']}] Score: {m['score']:.3f}\n   Chunk: {m['chunk']}")
                continue

            res = decision_engine.process_query(user_input)
            print(f"PRO AI > {res['response']}")
            print(f"         [Confidence: {res['confidence'] * 100:.1f}% | Source: {res['source']}]")

        except (KeyboardInterrupt, EOFError):
            print("\nBeende Python Mini AI Pro...")
            break

def main():
    display = os.environ.get("DISPLAY")
    if display or sys.platform in ["win32", "darwin"]:
        try:
            import tkinter as tk
            from python_mini_ai.ui.chat_window import ChatWindow
            root = tk.Tk()
            app = ChatWindow(root)
            root.title("Python Mini AI Pro - Deep Intelligence Suite")
            root.mainloop()
        except Exception as e:
            logger.warning(f"GUI konnte nicht geladen werden ({e}). Wechsle in den PRO CLI Modus.")
            run_pro_cli()
    else:
        run_pro_cli()

if __name__ == "__main__":
    main()
