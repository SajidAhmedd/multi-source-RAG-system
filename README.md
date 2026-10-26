---
title: Multi-Source RAG System
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
short_description: Ask questions across PDFs, CSVs and websites with citations
---

# Multi-Source RAG System

An enterprise knowledge-base assistant that answers questions across **PDFs, CSV files and web pages** at once,
with a source citation on every answer. Documents are chunked, embedded locally and indexed in FAISS; each
question goes through hybrid retrieval, deduplication and cross-encoder re-ranking before the LLM answers.

**Live demo: [rag.sajidahmed.dev](https://rag.sajidahmed.dev)** — click **Try Sample Data** and ask
*"What is the remote work policy?"* or *"Which smartphones cost less than $900?"*

## Features

- **Multi-source ingestion** — PDF pages, CSV rows (with row numbers) and cleaned web page text, all in one index
- **Local embeddings** — `all-MiniLM-L6-v2` (sentence-transformers) runs on CPU, so indexing costs nothing
- **FAISS vector store** built with LlamaIndex
- **Hybrid retrieval + re-ranking** — results are deduplicated and re-scored with the
  `ms-marco-MiniLM-L-6-v2` cross-encoder before being sent to the LLM
- **Cited answers** — every answer references its sources (`[Source: file]`, with page or row)
- **LLM fallback** — Gemini, Mistral and Groq are tried in order through one OpenAI-compatible client,
  so a rate-limited free tier doesn't break the app
- **Per-session indexes on the demo** — each visitor's uploads stay in their own session

## Architecture

```
 PDF / CSV / URL
        │
        ▼
  Ingestion  ──►  Processing  ──►  Chunking  ──►  Embeddings (MiniLM, local)  ──►  FAISS index
                  (clean, enrich                                                       │
                   metadata)                                                           │
                                                                                       ▼
  Question  ──►  Hybrid retrieval  ──►  Deduplicate  ──►  Cross-encoder re-rank  ──►  Top-k context
                                                                                       │
                                                                                       ▼
                                                  LLM (Gemini → Mistral → Groq fallback)
                                                                                       │
                                                                                       ▼
                                                                         Answer with citations
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| RAG framework | LlamaIndex |
| Vector store | FAISS |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Re-ranking | Cross-encoder `ms-marco-MiniLM-L-6-v2` |
| LLM | Google Gemini, Mistral, Groq (OpenAI-compatible APIs, automatic fallback) |
| Ingestion | pypdf, pandas, BeautifulSoup |
| UI | Streamlit |
| Deployment | Docker, Hugging Face Spaces, GitHub Actions |

## Quick Start

```bash
git clone https://github.com/SajidAhmedd/multi-source-RAG-system.git
cd multi-source-RAG-system
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt

cp .env.example .env         # add at least one free LLM key
streamlit run app.py
```

Free API keys: [Google AI Studio](https://aistudio.google.com/apikey) ·
[Mistral](https://console.mistral.ai) · [Groq](https://console.groq.com)

On Windows, `run.bat` creates the virtual environment, installs dependencies and starts the app.

### With Docker

```bash
docker build -t multi-source-rag .
docker run -p 7860:7860 --env-file .env multi-source-rag
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDERS` | `gemini,mistral,groq` | Order in which LLM providers are tried |
| `GEMINI_API_KEY` / `MISTRAL_API_KEY` / `GROQ_API_KEY` | — | Provider keys; set at least one |
| `GEMINI_MODEL` / `MISTRAL_MODEL` / `GROQ_MODEL` | `gemini-3-flash-preview` / `mistral-small-latest` / `openai/gpt-oss-20b` | Model per provider |
| `PERSIST_INDEX` | `true` (`false` in Docker) | Save the uploaded documents' index to `index_storage/` |

Chunk size, overlap and retrieval depth are in `src/config.py`.

## Project Structure

```
multi-source-RAG-system/
├── app.py                    # Streamlit UI
├── data/                     # Sample CSVs (company policies, product catalog)
├── src/
│   ├── config.py             # Settings
│   ├── ingestion/            # PDF, CSV and web loaders
│   ├── processing/           # Cleaning, metadata enrichment, chunking
│   ├── indexing/             # FAISS index + local embeddings
│   ├── retrieval/            # Hybrid retriever and cross-encoder re-ranker
│   └── llm/                  # Answer generation with citations, multi-provider LLM client
├── Dockerfile
└── requirements.txt
```

## Deployment

The live demo runs as a Docker container on Hugging Face Spaces. Every push to `main` is deployed by
`.github/workflows/deploy-hf.yml`, which also pings the demo twice a day to keep the free Space awake.
`rag.sajidahmed.dev` is served through a Cloudflare Worker that proxies the Space.

## License

MIT License — see [LICENSE](LICENSE).
