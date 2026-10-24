# Multi-Source RAG System - Streamlit app
# Runs on Hugging Face Spaces (port 7860) or anywhere Docker runs.

FROM python:3.11-slim

WORKDIR /app

# Non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser

# CPU-only PyTorch keeps the image small (sentence-transformers needs torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PERSIST_INDEX=false

# Download the embedding and re-ranking models at build time so cold starts are fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-7860} --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"]
