import os
from pathlib import Path
from typing import Dict, Any

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Data and Model Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
TRAINING_DIR = BASE_DIR / "training"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "model_weights.npz"
DB_PATH = DATA_DIR / "memory.db"
KNOWLEDGE_PATH = KNOWLEDGE_DIR / "knowledge.json"
TRAINING_DATA_PATH = TRAINING_DIR / "training_data.json"

# Neural Network Hyperparameters
HIDDEN_DIMS = [64, 32]
LEARNING_RATE = 0.01
EPOCHS = 300
BATCH_SIZE = 8

# Confidence Thresholds
CONFIDENCE_THRESHOLD_HIGH = 0.80
CONFIDENCE_THRESHOLD_MEDIUM = 0.50
CONFIDENCE_THRESHOLD_LOW = 0.30

# Memory Settings
SHORT_TERM_MEMORY_SIZE = 10

# Logging
LOG_LEVEL = "INFO"
DEBUG_MODE = True
