import os
from pathlib import Path
from typing import List, Dict, Any

# Base Directory for Python Mini AI Pro
BASE_DIR = Path(__file__).resolve().parent

# Directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
TRAINING_DIR = BASE_DIR / "training"
PLUGINS_DIR = BASE_DIR / "plugins"

for d in [DATA_DIR, MODELS_DIR, KNOWLEDGE_DIR, TRAINING_DIR, PLUGINS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "pro_model_weights.npz"
DB_PATH = DATA_DIR / "pro_memory.db"
VECTOR_STORE_PATH = DATA_DIR / "vector_store.json"
KNOWLEDGE_PATH = KNOWLEDGE_DIR / "pro_knowledge.json"
TRAINING_DATA_PATH = TRAINING_DIR / "pro_training_data.json"

# Pro Model Hyperparameters
HIDDEN_DIMS = [128, 64, 32]
LEARNING_RATE = 0.005
EPOCHS = 500
BATCH_SIZE = 16

# Decision Thresholds
CONFIDENCE_THRESHOLD_HIGH = 0.85
CONFIDENCE_THRESHOLD_MEDIUM = 0.55
CONFIDENCE_THRESHOLD_LOW = 0.35

# Short-Term Memory Buffer Size
SHORT_TERM_MEMORY_SIZE = 25

# Code Execution Sandbox Timeout
SANDBOX_TIMEOUT_SECONDS = 3.0

DEBUG_MODE = True
