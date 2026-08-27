import logging
from typing import Dict, Any, Optional
from python_mini_ai.ai.intent_classifier import IntentClassifier
from python_mini_ai.ai.response_generator import ResponseGenerator
from python_mini_ai.nlp.similarity import SimilarityCalculator
from python_mini_ai.config import (
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_MEDIUM,
    CONFIDENCE_THRESHOLD_LOW
)

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Combines Knowledge Base, Intent Classification (Neural Network), Rule-based matching,
    and Chat Memory into a prioritized final answer with transparency scores.
    """

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        response_generator: ResponseGenerator,
        knowledge_base: Any = None,
        memory_manager: Any = None
    ):
        self.intent_classifier = intent_classifier
        self.response_generator = response_generator
        self.knowledge_base = knowledge_base
        self.memory_manager = memory_manager
        self.similarity_calculator = SimilarityCalculator()

    def process_query(self, user_text: str) -> Dict[str, Any]:
        """
        Evaluates input text through:
        1. Exact/High Knowledge Base Match
        2. Neural Network Intent Match
        3. Rule / Pattern Matching
        4. Memory context analysis
        5. Fallback response
        """
        # Step 1: Check Knowledge Base
        kb_match = None
        kb_score = 0.0
        if self.knowledge_base:
            kb_result = self.knowledge_base.search(user_text)
            if kb_result:
                kb_match, kb_score = kb_result["entry"], kb_result["score"]

        if kb_match and kb_score >= CONFIDENCE_THRESHOLD_HIGH:
            debug_info = {
                "input": user_text,
                "response_source": "knowledge_base",
                "confidence": kb_score,
                "kb_score": kb_score,
                "detected_intent": "knowledge_match",
                "nn_confidence": 0.0
            }
            return {
                "response": kb_match["answer"],
                "confidence": kb_score,
                "source": "knowledge_base",
                "debug": debug_info
            }

        # Step 2: Neural Network Intent Classifier
        nn_result = self.intent_classifier.classify(user_text)
        detected_intent = nn_result["intent"]
        nn_confidence = nn_result["confidence"]

        # Step 3: Check learned memory questions/answers if present
        learned_match = None
        learned_score = 0.0
        if self.memory_manager and hasattr(self.memory_manager, "search_long_term_qa"):
            learned_qa = self.memory_manager.search_long_term_qa(user_text)
            if learned_qa:
                learned_match, learned_score = learned_qa["answer"], learned_qa["score"]

        if learned_match and learned_score >= CONFIDENCE_THRESHOLD_HIGH:
            debug_info = {
                "input": user_text,
                "response_source": "long_term_memory",
                "confidence": learned_score,
                "kb_score": kb_score,
                "detected_intent": detected_intent,
                "nn_confidence": nn_confidence
            }
            return {
                "response": learned_match,
                "confidence": learned_score,
                "source": "long_term_memory",
                "debug": debug_info
            }

        # Decide between KB (medium match) vs NN vs Default
        if kb_match and kb_score >= CONFIDENCE_THRESHOLD_MEDIUM and kb_score > nn_confidence:
            response_text = kb_match["answer"]
            final_confidence = kb_score
            source = "knowledge_base"
        elif nn_confidence >= CONFIDENCE_THRESHOLD_MEDIUM and detected_intent != "unknown":
            response_text = self.response_generator.generate_response(detected_intent)
            final_confidence = nn_confidence
            source = "neural_network"
        elif kb_match and kb_score >= CONFIDENCE_THRESHOLD_LOW:
            response_text = f"Ich bin mir nicht ganz sicher, aber meinst du: '{kb_match['question']}'? {kb_match['answer']}"
            final_confidence = kb_score
            source = "knowledge_base_low_confidence"
        elif nn_confidence >= CONFIDENCE_THRESHOLD_LOW and detected_intent != "unknown":
            response_text = self.response_generator.generate_response(detected_intent)
            final_confidence = nn_confidence
            source = "neural_network_low_confidence"
        else:
            response_text = self.response_generator.fallback_response()
            final_confidence = max(nn_confidence, kb_score, 0.20)
            source = "fallback"

        debug_info = {
            "input": user_text,
            "response_source": source,
            "confidence": final_confidence,
            "kb_score": kb_score,
            "detected_intent": detected_intent,
            "nn_confidence": nn_confidence
        }

        # Save to short term memory if available
        if self.memory_manager:
            self.memory_manager.add_short_term("user", user_text)
            self.memory_manager.add_short_term("assistant", response_text)

        return {
            "response": response_text,
            "confidence": final_confidence,
            "source": source,
            "debug": debug_info
        }
