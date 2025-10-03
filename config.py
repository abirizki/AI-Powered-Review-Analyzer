import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Database
DATABASE_PATH = os.path.join(DATA_PROCESSED, "reviews_analyzer.db")

# AI Model settings
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

# Ensure directories exist
for directory in [DATA_RAW, DATA_PROCESSED, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

print("✅ Config loaded successfully!")
