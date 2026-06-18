"""Configuration settings for the Multi-Source RAG system."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Settings (for google-genai package)
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Chunking Settings
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# Retrieval Settings
TOP_K_RESULTS = 15
SIMILARITY_THRESHOLD = 0.5

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index_storage")
