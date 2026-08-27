import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
TRAINING_DIR = BASE_DIR / "training"

for d in [DATA_DIR, MODELS_DIR, KNOWLEDGE_DIR, TRAINING_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "ultra_ensemble_weights.npz"
DB_PATH = DATA_DIR / "ultra_memory.db"
VECTOR_STORE_PATH = DATA_DIR / "ultra_vector_store.json"
KNOWLEDGE_PATH = KNOWLEDGE_DIR / "ultra_knowledge.json"
TRAINING_DATA_PATH = TRAINING_DIR / "ultra_training_data.json"

HIDDEN_DIMS = [128, 64, 32]
LEARNING_RATE = 0.005
EPOCHS = 300
BATCH_SIZE = 16

CONFIDENCE_THRESHOLD_HIGH = 0.85
CONFIDENCE_THRESHOLD_MEDIUM = 0.55
CONFIDENCE_THRESHOLD_LOW = 0.35

SHORT_TERM_MEMORY_SIZE = 30
