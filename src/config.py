"""Configuration settings for the Multi-Source RAG system."""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM providers, tried in this order (only those with an API key are used).
# If one is rate-limited or down, the request falls back to the next - see src/llm/llm_client.py.
# Free keys: https://aistudio.google.com/apikey  https://console.mistral.ai  https://console.groq.com
LLM_PROVIDERS = os.getenv("LLM_PROVIDERS", "gemini,mistral,groq")

# Embeddings run locally with sentence-transformers (see src/indexing/vector_store.py)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking Settings
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# Retrieval Settings
TOP_K_RESULTS = 15
SIMILARITY_THRESHOLD = 0.5

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index_storage")

# Save uploaded documents' index to INDEX_DIR. Disabled on the public demo so each
# visitor's uploads stay in their own session instead of replacing everyone's index.
PERSIST_INDEX = os.getenv("PERSIST_INDEX", "true").lower() == "true"
