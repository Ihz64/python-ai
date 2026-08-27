import logging
from typing import Dict, Any, Tuple, Optional
from python_mini_ai.training.dataset import DatasetManager
from python_mini_ai.ai.neural_network import NeuralNetwork
from python_mini_ai.memory.memory_manager import MemoryManager
from python_mini_ai.knowledge.knowledge_base import KnowledgeBase
from python_mini_ai import config

logger = logging.getLogger(__name__)

class LearningSystem:
    """
    Handles active user feedback (+/-), dataset updating, persistent Q&A learning,
    and automatic or manual re-training of the neural network model.
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        memory_manager: MemoryManager,
        knowledge_base: KnowledgeBase,
        model: NeuralNetwork
    ):
        self.dataset_manager = dataset_manager
        self.memory_manager = memory_manager
        self.knowledge_base = knowledge_base
        self.model = model

    def record_feedback(
        self,
        user_query: str,
        ai_response: str,
        is_positive: bool,
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes feedback for the last chat turn:
        - Positive feedback: Stores user query & answer in long-term memory and optional intent training.
        - Negative feedback: Flags answer and allows learning corrections.
        """
        if is_positive:
            # Store high-confidence Q&A in long term memory
            self.memory_manager.add_learned_fact(user_query, ai_response, score=1.0)

            # Add pattern example to dataset manager if tag provided
            if tag:
                self.dataset_manager.add_example(tag=tag, pattern=user_query, response=ai_response)

            return {
                "status": "success",
                "message": "Positives Feedback gespeichert! Die KI hat diese Antwort verinnerlicht.",
                "retrained": False
            }
        else:
            return {
                "status": "acknowledged",
                "message": "Negatives Feedback registriert. Du kannst eine Korrektur im Trainingsbereich hinzufügen.",
                "retrained": False
            }

    def teach_new_qa(self, question: str, answer: str, tag: str = "custom_learning") -> Tuple[bool, str]:
        """
        Directly teaches the AI a new Question-Answer pair:
        1. Adds to DatasetManager JSON
        2. Adds to KnowledgeBase JSON
        3. Adds to SQLite Long-Term Memory
        4. Re-trains Neural Network weights and saves model
        """
        try:
            from python_mini_ai.ai.trainer import ModelTrainer

            # 1. Dataset
            self.dataset_manager.add_example(tag=tag, pattern=question, response=answer)

            # 2. Knowledge Base
            self.knowledge_base.add_entry(topic="Gelernte Fragen", question=question, answer=answer, tags=[tag])

            # 3. Long-term memory
            self.memory_manager.add_learned_fact(question, answer, score=1.0)

            # 4. Retrain model
            trainer = ModelTrainer(self.dataset_manager, hidden_dims=config.HIDDEN_DIMS, lr=config.LEARNING_RATE)
            new_model, loss_history = trainer.train(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

            # Copy new weights
            self.model.weights = new_model.weights
            self.model.biases = new_model.biases
            self.model.layer_sizes = new_model.layer_sizes
            self.model.save(config.MODEL_PATH)

            return True, f"Erfolgreich gelernt! Modell neu trainiert. Finaler Loss: {loss_history[-1]:.4f}"
        except Exception as e:
            logger.error(f"Error teaching new QA: {e}")
            return False, f"Fehler beim Lernen: {str(e)}"

    def retrain_model(self) -> Tuple[bool, str]:
        """Triggers full re-training on current training_data.json dataset."""
        try:
            from python_mini_ai.ai.trainer import ModelTrainer

            trainer = ModelTrainer(self.dataset_manager, hidden_dims=config.HIDDEN_DIMS, lr=config.LEARNING_RATE)
            new_model, loss_history = trainer.train(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

            self.model.weights = new_model.weights
            self.model.biases = new_model.biases
            self.model.layer_sizes = new_model.layer_sizes
            self.model.save(config.MODEL_PATH)

            return True, f"Modell erfolgreich neu trainiert. Finaler Loss: {loss_history[-1]:.4f}"
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return False, f"Fehler beim Re-Training: {str(e)}"
