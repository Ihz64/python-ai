import sys
import os
import logging
from pathlib import Path
import numpy as np

from python_mini_ai_ultra import config
from python_mini_ai_ultra.knowledge.dataset_generator import generate_ultra_dataset
from python_mini_ai_ultra.ai.ensemble_network import UltraEnsembleNeuralNetwork
from python_mini_ai_ultra.ai.trainer import UltraModelTrainer
from python_mini_ai.ai.math_engine import MathEngine
from python_mini_ai_pro.rag.vector_store import VectorStoreRAG
from python_mini_ai_pro.sandbox.code_sandbox import CodeSandbox
from python_mini_ai.training.dataset import DatasetManager
from python_mini_ai.knowledge.knowledge_base import KnowledgeBase
from python_mini_ai.memory.memory_manager import MemoryManager
from python_mini_ai.ai.inference import InferenceEngine
from python_mini_ai.ai.intent_classifier import IntentClassifier
from python_mini_ai.ai.response_generator import ResponseGenerator
from python_mini_ai.ai.decision_engine import DecisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PythonMiniAIUltra")

def run_ultra_cli():
    print("=" * 75)
    print("   🌌 PYTHON MINI AI ULTRA - HIGH-CAPACITY LOCAL INTELLIGENCE v3.0   ")
    print("=" * 75)
    print("Prüfe und initialisiere Ultra-Wissensbasis & Multi-Architecture Ensemble...\n")

    if not config.KNOWLEDGE_PATH.exists() or not config.TRAINING_DATA_PATH.exists():
        print("Erstelle Ultra Dataset...")
        generate_ultra_dataset(config.KNOWLEDGE_PATH, config.TRAINING_DATA_PATH)

    dataset_manager = DatasetManager(config.TRAINING_DATA_PATH)
    knowledge_base = KnowledgeBase(config.KNOWLEDGE_PATH)
    memory_manager = MemoryManager(config.DB_PATH, max_short_term=config.SHORT_TERM_MEMORY_SIZE)
    vector_store = VectorStoreRAG(config.VECTOR_STORE_PATH)
    sandbox = CodeSandbox()

    # Pre-index Knowledge Base into VectorStore
    for idx, entry in enumerate(knowledge_base.entries[:30]):
        vector_store.add_document(
            doc_id=f"doc_{idx}_{entry.get('topic', 'doc')}",
            title=entry.get("question", "Titel"),
            content=entry.get("answer", "")
        )

    vocab_size = len(dataset_manager.vocabulary) if dataset_manager.vocabulary else 10
    num_classes = len(dataset_manager.classes) if dataset_manager.classes else 3
    layer_sizes = [vocab_size] + config.HIDDEN_DIMS + [num_classes]

    ensemble_net = UltraEnsembleNeuralNetwork(layer_sizes=layer_sizes)
    if config.MODEL_PATH.exists():
        ensemble_net.load(config.MODEL_PATH)
    else:
        print("Trainiere Ultra Ensemble Neural Network...")
        trainer = UltraModelTrainer(dataset_manager, hidden_dims=config.HIDDEN_DIMS, lr=config.LEARNING_RATE)
        ensemble_net, loss_hist = trainer.train(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
        ensemble_net.save(config.MODEL_PATH)
        print("Ultra Ensemble Modell erfolgreich trainiert!")

    inference_engine = InferenceEngine(ensemble_net, dataset_manager)
    intent_classifier = IntentClassifier(inference_engine)
    response_generator = ResponseGenerator(dataset_manager)
    decision_engine = DecisionEngine(
        intent_classifier=intent_classifier,
        response_generator=response_generator,
        knowledge_base=knowledge_base,
        memory_manager=memory_manager,
        math_engine=MathEngine()
    )

    print("\nULTRA KI BEREIT!")
    print("Befehle: 'train' (Lokales Re-Training), 'run <code>' (Sandbox), 'search <query>' (RAG), 'exit' (Beenden).\n")

    while True:
        try:
            user_input = input("\nULTRA-AI User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "beenden"]:
                print("Auf Wiedersehen von Python Mini AI Ultra!")
                break

            if user_input.lower() == "train":
                print("Starte lokales Training des Ultra Neural Ensembles...")
                trainer = UltraModelTrainer(dataset_manager, hidden_dims=config.HIDDEN_DIMS, lr=config.LEARNING_RATE)
                ensemble_net, loss_hist = trainer.train(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
                ensemble_net.save(config.MODEL_PATH)
                print(f"Lokales Training abgeschlossen! Finaler Loss: {loss_hist[-1]:.4f}")
                continue

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
                print(f"\n[ULTRA RAG RETRIEVAL]")
                for idx, m in enumerate(rag_matches, 1):
                    print(f"{idx}. [{m['title']}] Score: {m['score']:.3f}\n   Chunk: {m['chunk']}")
                continue

            res = decision_engine.process_query(user_input)
            print(f"ULTRA AI > {res['response']}")
            print(f"           [Confidence: {res['confidence'] * 100:.1f}% | Source: {res['source']}]")

        except (KeyboardInterrupt, EOFError):
            print("\nBeende Python Mini AI Ultra...")
            break

def main():
    run_ultra_cli()

if __name__ == "__main__":
    main()
