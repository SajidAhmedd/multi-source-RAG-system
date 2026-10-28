# Multi-Source RAG System: A Beginner's Tutorial

This tutorial walks you through the whole code base of the Multi-Source RAG System, file by file and function by function. By the end you should be able to explain what every part does, why it was written that way, and where you would change it.

You need mid-level Python: functions, classes, imports and `try/except`. You do **not** need to know web apps, vector databases or AI libraries. Each one is explained when it first appears.

## Contents

1. [How to use this tutorial](#1-how-to-use-this-tutorial)
2. [The big picture](#2-the-big-picture)
3. [Folder map](#3-folder-map)
4. [Python concepts you need](#4-python-concepts-you-need)
5. [The libraries](#5-the-libraries)
6. [Every file and every function](#6-every-file-and-every-function)
7. [One question traced from start to finish](#7-one-question-traced-from-start-to-finish)
8. [Things to notice](#8-things-to-notice)
9. [Exercises](#9-exercises)
10. [Glossary](#10-glossary)

---

## 1. How to use this tutorial

This tutorial is meant to be read **next to the code**, not on its own.

1. Open the repo folder in VS Code (`File → Open Folder…`).
2. Open `TUTORIAL.md` and press **Ctrl+K V** (Cmd+K V on a Mac). The Markdown preview opens in a side panel, so the tutorial sits on the right and the code on the left.
3. Every link like [src/llm/generator.py:113-148](src/llm/generator.py#L113-L148) is clickable. In the preview, clicking it opens that file at those lines. Read the explanation, then look left at the real code.
4. Links to sections of this file (like the contents list above) jump inside the tutorial.

Why read this way? Code is only understood by reading it. The tutorial tells you *what to look for* and *why it's there*; the code shows you *exactly how*. Snippets below are short extracts. The link always leads to the full version.

Read sections 2 to 5 first. Section 6 follows the order in which files import each other, so each file is explained only after the files it depends on. The project is small (about 1,300 lines of Python), so you can realistically read every line.

---

## 2. The big picture

### What the project does

You give the app some documents (PDF files, CSV spreadsheets, or web page addresses) and then ask questions in a chat box: *"What is the remote work policy?"*, *"Which smartphones cost less than $900?"*. The app answers in plain English and tells you **which file, page or row** each fact came from.

An LLM (large language model, like Gemini) can write good answers, but it doesn't know *your* documents. You also can't paste 500 pages into every question. The standard solution is **RAG**, retrieval-augmented generation:

1. **Index once.** Split all documents into small pieces (*chunks*), turn each chunk into a list of numbers that captures its meaning (an *embedding*), and store those numbers in a *vector index*.
2. **For each question:** turn the question into numbers the same way, find the chunks whose numbers are closest (**retrieval**), and send only those chunks plus the question to the LLM (**generation**), with instructions to answer from them and cite them.

This project adds two extra steps between retrieval and generation: **deduplication** (drop near-identical chunks) and **re-ranking** (score each chunk again with a more accurate model and keep the best).

### How the pieces fit together

```
 INDEXING (when you click "Process & Index" or "Try Sample Data")

   PDF files ──► pdf_loader ─┐
   CSV files ──► csv_loader ─┼─► Documents ─► processor ─► chunker ─► nodes
   URLs ──────► web_loader ──┘   (src/ingestion)  (clean,     (split into
                                                   metadata)    chunks)
                                                                   │
                                                                   ▼
                                   LocalEmbedding (all-MiniLM-L6-v2, on your CPU)
                                                                   │
                                                                   ▼
                                       FAISS index (in memory, optionally saved
                                       to index_storage/)   src/indexing/vector_store.py


 ANSWERING (every chat message)          src/llm/generator.py  RAGGenerator.query()

   question ─► MultiRetriever.hybrid_retrieve ─► deduplicate ─► cross-encoder re-rank
               (up to 60 nearest)                (drop          (keep best 15)
               src/retrieval/retriever.py         near-copies)   src/retrieval/reranker.py
                                                                         │
                                                                         ▼
                                     prompt = instructions + 15 chunks + question
                                                                         │
                                                                         ▼
                                  LLMClient: Gemini ─(fails?)─► Mistral ─► Groq
                                  src/llm/llm_client.py
                                                                         │
                                                                         ▼
                                  answer + list of sources ─► Streamlit chat (app.py)
```

Things to notice already:

- **Embeddings and re-ranking run locally** on your CPU (free, private). Only the final answer uses a paid-or-free-tier online LLM.
- **One process.** Streamlit runs `app.py`, which calls the `src` package directly. There is no separate server.
- **Each browser session has its own index** in `st.session_state`, except the sample index, which is built once and shared.

---

## 3. Folder map

Every file in the repo, one line each.

```
multi-source-RAG-system/
├── .dockerignore              files Docker should not copy into the image (.env, index_storage/…)
├── .env.example               template for your keys and settings; copy it to .env
├── .github/workflows/
│   └── deploy-hf.yml          GitHub Actions: push to Hugging Face Spaces, ping the demo twice a day
├── .gitignore                 files git should not track (.env, index_storage/, caches)
├── Dockerfile                 container recipe: CPU-only PyTorch, pre-downloaded models, Streamlit on 7860
├── LICENSE                    MIT licence
├── README.md                  overview; the block at the top is also the Hugging Face Space config
├── app.py                     the Streamlit web app: sidebar for data, chat for questions
├── data/
│   ├── company_policies.csv   8 sample HR/IT/legal policies
│   └── sample_products.csv    10 sample products with price and stock
├── requirements.txt           Python packages to install
├── run.bat                    Windows helper: create venv, install, start the app
└── src/
    ├── __init__.py            one-line docstring; makes src/ a package
    ├── config.py              settings: provider order, chunk sizes, top-k, folders
    ├── indexing/
    │   ├── __init__.py        re-exports create_index, load_index, add_documents_to_index
    │   └── vector_store.py    local embedding model + FAISS index create/load/add
    ├── ingestion/
    │   ├── __init__.py        re-exports load_pdfs, load_csvs, load_urls
    │   ├── csv_loader.py      one Document per CSV row
    │   ├── pdf_loader.py      one Document per PDF page
    │   └── web_loader.py      one Document per web page (HTML → text)
    ├── llm/
    │   ├── __init__.py        re-exports RAGGenerator
    │   ├── generator.py       retrieval → dedupe → re-rank → prompt → LLM → answer + sources
    │   └── llm_client.py      one LLM client that falls back across Gemini, Mistral and Groq
    ├── processing/
    │   ├── __init__.py        re-exports chunk_documents, process_documents
    │   ├── chunker.py         splits Documents into chunks (nodes)
    │   └── processor.py       cleans text and adds statistics to metadata
    └── retrieval/
        ├── __init__.py        re-exports MultiRetriever, rerank_results
        ├── reranker.py        cross-encoder re-ranking and simple deduplication
        └── retriever.py       vector retrieval (with optional filtering)
```

Created at run time (ignored by git): `index_storage/` (a saved index, when `PERSIST_INDEX=true`).

---

## 4. Python concepts you need

These are the Python features this code uses beyond "functions, classes, imports, try/except". One-line recap each, with an example location.

**Packages and `__init__.py`.** A folder with an `__init__.py` is a *package*. Code in `__init__.py` runs when the package is first imported. Here each one re-exports the useful names, so `from src.ingestion import load_pdfs` works instead of `from src.ingestion.pdf_loader import load_pdfs`. [src/ingestion/\_\_init\_\_.py:1-7](src/ingestion/__init__.py#L1-L7). `__all__` lists the public names.

**Relative imports.** `from ..config import INDEX_DIR` means "from the `config` module one package up". One dot is the current package, two dots its parent. [src/indexing/vector_store.py:16](src/indexing/vector_store.py#L16). They only work when the file is imported as part of the package (as `app.py` does with `from src...`), not when run as a script.

**Type hints, `Optional`, `List`, `Tuple`.** `def query(self, question: str) -> Tuple[str, List[dict]]` documents that it returns a pair (answer, list of dicts). `Optional[int]` means "int or None". [src/llm/generator.py:113](src/llm/generator.py#L113).

**Returning a tuple and unpacking it.** `return response, sources` returns two values; `answer, sources = generator.query(q)` receives them. [app.py:244](app.py#L244).

**`x = a or b`.** Gives `a` unless it's "falsy" (None, 0, "", empty list), otherwise `b`. Used for defaults: `persist_dir = persist_dir or INDEX_DIR`. [src/indexing/vector_store.py:69](src/indexing/vector_store.py#L69). Also chained: `metadata.get("file_name") or metadata.get("source") or ...`. [src/llm/generator.py:68](src/llm/generator.py#L68).

**`**kwargs`.** Collects extra keyword arguments and passes them on: `load_csvs(files, max_rows=100)` forwards `max_rows` to `load_csv`. [src/ingestion/csv_loader.py:89-95](src/ingestion/csv_loader.py#L89-L95).

**Inheritance and abstract methods.** `class LocalEmbedding(BaseEmbedding)` fills in the methods LlamaIndex's base class requires (`_get_query_embedding`, `_get_text_embedding`…). LlamaIndex then calls them when it needs embeddings. [src/indexing/vector_store.py:19-48](src/indexing/vector_store.py#L19-L48).

**`async def`.** An asynchronous version of a method. The base class requires these too; here they just call the normal version. [src/indexing/vector_store.py:39-43](src/indexing/vector_store.py#L39-L43).

**`lambda` as a sort key.** `sorted(results, key=lambda x: x.score, reverse=True)` sorts objects by their `score`, highest first. [src/retrieval/reranker.py:53](src/retrieval/reranker.py#L53).

**Sets for overlap.** `set(a.split()) & set(b.split())` gives the words two texts share. [src/retrieval/reranker.py:104](src/retrieval/reranker.py#L104).

**`with` blocks (context managers).** `with tempfile.NamedTemporaryFile(...) as tmp:` opens a file and closes it when the block ends. [app.py:93-95](app.py#L93-L95).

**The walrus operator `:=`.** `if query := st.chat_input(...):` assigns and tests in one step: "store the input in `query`, and continue if it isn't empty". [app.py:233](app.py#L233).

**Decorators.** `@st.cache_resource` wraps a function so its result is computed once and reused. [app.py:45-49](app.py#L45-L49). `@dataclass` and `@property` appear in `llm_client.py`: [src/llm/llm_client.py:26-46](src/llm/llm_client.py#L26-L46).

**`types.SimpleNamespace`.** A blank object you can attach attributes to; used to build the `client.chat.completions` shape. [src/llm/llm_client.py:137](src/llm/llm_client.py#L137).

**Regular expressions.** `re.sub(r'\s+', ' ', text)` replaces every run of whitespace with one space. [src/ingestion/web_loader.py:13](src/ingestion/web_loader.py#L13).

**`str.format` templates.** `"Question: {query_str}".format(query_str=q)` fills named placeholders. [src/llm/generator.py:142](src/llm/generator.py#L142).

---

## 5. The libraries

All the packages in [requirements.txt](requirements.txt), in that file's order.

**LlamaIndex** (`llama-index`, imported as `llama_index.core`). A framework for building RAG apps. It gives you the building blocks:
- `Document`: a piece of source text plus a `metadata` dict.
- *Node* (`TextNode`): a chunk of a Document, the unit that gets embedded and retrieved.
- Node parsers such as `SentenceSplitter`: turn Documents into nodes.
- `VectorStoreIndex`: embeds nodes and stores them in a vector store.
- `VectorIndexRetriever`: finds the nodes most similar to a query, returned as `NodeWithScore` (a node plus a `score`).
- `Settings`: global defaults, e.g. which embedding model to use.

```python
from llama_index.core import Document, VectorStoreIndex
docs = [Document(text="Refunds take 5-7 days.", metadata={"file_name": "policy.txt"})]
index = VectorStoreIndex.from_documents(docs, embed_model=my_embed_model)
for hit in index.as_retriever(similarity_top_k=1).retrieve("how long for refunds?"):
    print(hit.score, hit.node.get_content())
```
*Why here:* it handles chunking, indexing and retrieval so the project can focus on the multi-source parts.
*Where:* loaders, [src/processing/chunker.py](src/processing/chunker.py), [src/indexing/vector_store.py](src/indexing/vector_store.py), [src/retrieval/retriever.py](src/retrieval/retriever.py).
Note that this project uses LlamaIndex for retrieval but **not** for the LLM call; that goes through its own `LLMClient`.

**llama-index-vector-stores-faiss** (`llama-index-vector-stores-faiss`). The plug that lets LlamaIndex store vectors in FAISS: `FaissVectorStore`.
*Where:* [src/indexing/vector_store.py:14](src/indexing/vector_store.py#L14), [src/indexing/vector_store.py:73](src/indexing/vector_store.py#L73).

**OpenAI SDK** (`openai`). The official client for OpenAI's chat API. Gemini, Mistral and Groq offer **OpenAI-compatible** endpoints (same request format, different address), so this one library talks to all three by changing `base_url` and `api_key`.

```python
from openai import OpenAI
c = OpenAI(api_key="...", base_url="https://api.mistral.ai/v1")
r = c.chat.completions.create(model="mistral-small-latest",
        messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content)
```
*Where:* [src/llm/llm_client.py:23](src/llm/llm_client.py#L23), [src/llm/llm_client.py:127](src/llm/llm_client.py#L127).

**pypdf** (`pypdf`). Reads PDF files: `PdfReader(path).pages`, then `page.extract_text()`. Scanned PDFs (pictures of text) give no text, because there is no OCR.
*Where:* [src/ingestion/pdf_loader.py:14-18](src/ingestion/pdf_loader.py#L14-L18).

**Beautiful Soup** (`beautifulsoup4`, imported as `bs4`). Parses HTML so you can find tags and pull out text.

```python
from bs4 import BeautifulSoup
soup = BeautifulSoup("<html><title>Hi</title><p>Text</p></html>", "html.parser")
print(soup.find("title").get_text(), soup.get_text())
```
*Where:* [src/ingestion/web_loader.py:30-47](src/ingestion/web_loader.py#L30-L47).

**requests** (`requests`). Makes HTTP calls; used to download web pages.
*Where:* [src/ingestion/web_loader.py:27-28](src/ingestion/web_loader.py#L27-L28).

**pandas** (`pandas`). The standard table library. `pd.read_csv` gives a DataFrame; `df.iterrows()` loops over rows as `(index, row)` pairs; `pd.notna(x)` is true if a cell isn't empty.
*Where:* [src/ingestion/csv_loader.py:40-57](src/ingestion/csv_loader.py#L40-L57).

**FAISS** (`faiss-cpu`). Facebook AI Similarity Search, a fast library for finding nearest vectors. `IndexFlatL2(384)` is the simplest kind: it stores every 384-number vector and, for a query, measures the straight-line (L2) **distance** to all of them. Smaller distance = more similar.

```python
import faiss, numpy as np
idx = faiss.IndexFlatL2(3)
idx.add(np.array([[0,0,0],[1,1,1]], dtype="float32"))
print(idx.search(np.array([[0.9,1,1]], dtype="float32"), 1))   # distance, position
```
*Where:* [src/indexing/vector_store.py:72](src/indexing/vector_store.py#L72).

**Streamlit** (`streamlit`). Builds a web page from a Python script. **The whole script re-runs on every interaction**; values that must survive go in `st.session_state`. `st.chat_input` and `st.chat_message` give a chat layout; `st.file_uploader` gives upload boxes; `st.rerun()` forces an immediate re-run; `@st.cache_resource` keeps one shared object (like a model or index) across re-runs and sessions.
*Where:* [app.py](app.py).

**python-dotenv** (`python-dotenv`). `load_dotenv()` copies `KEY=value` lines from `.env` into environment variables, where `os.getenv` reads them. Keeps keys out of the code.
*Where:* [src/config.py:4-6](src/config.py#L4-L6), [app.py:10-13](app.py#L10-L13).

**tqdm** (`tqdm`) and **NumPy** (`numpy`). tqdm draws progress bars; NumPy handles number arrays. Neither is imported by this project's code; both are used inside other libraries (sentence-transformers, FAISS, LlamaIndex). They're listed to pin minimum versions.

**sentence-transformers** (`sentence-transformers`). Runs small transformer models locally (it needs PyTorch). Two kinds are used:
- A **bi-encoder** (`SentenceTransformer("all-MiniLM-L6-v2")`) turns *one* text into a vector of 384 numbers. Fast, so it can embed thousands of chunks in advance.
- A **cross-encoder** (`CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`) reads the question and a chunk *together* and outputs one relevance score. More accurate, because it sees both texts at once, but it must run once per (question, chunk) pair, so it's only used on a short list.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
vec = SentenceTransformer("all-MiniLM-L6-v2").encode("hello")        # shape (384,)
score = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict([("q", "doc")])
```
*Where:* [src/indexing/vector_store.py:24-25](src/indexing/vector_store.py#L24-L25), [src/retrieval/reranker.py:8](src/retrieval/reranker.py#L8), [src/retrieval/reranker.py:21](src/retrieval/reranker.py#L21).

---

## 6. Every file and every function

Files are in **dependency order**: each file appears only after the project files it imports.

### 6.1 Settings: src/config.py

Imports: no project files.

[src/config.py:1-30](src/config.py#L1-L30) is a plain module of constants. `load_dotenv()` runs on import ([line 6](src/config.py#L6)).

| Name | Value | Used? |
|---|---|---|
| `LLM_PROVIDERS` ([line 11](src/config.py#L11)) | env, default `"gemini,mistral,groq"` | yes, by the app and generator |
| `EMBEDDING_MODEL` ([line 14](src/config.py#L14)) | `"all-MiniLM-L6-v2"` | **no**: the model name is hard-coded again in `vector_store.py` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` ([lines 17-18](src/config.py#L17-L18)) | 1024, 200 | **no**: `chunker.py` has the same numbers as defaults |
| `TOP_K_RESULTS` ([line 21](src/config.py#L21)) | 15 | yes |
| `SIMILARITY_THRESHOLD` ([line 22](src/config.py#L22)) | 0.5 | **no** |
| `DATA_DIR`, `INDEX_DIR` ([lines 25-26](src/config.py#L25-L26)) | `data/`, `index_storage/` next to `src/` | yes |
| `PERSIST_INDEX` ([line 30](src/config.py#L30)) | env, default `true` | yes |

`DATA_DIR` is built from `__file__` (this file's path), so it works no matter which folder you start Python from. `PERSIST_INDEX` converts the text `"true"`/`"false"` to a boolean with `.lower() == "true"`.

*Why does `PERSIST_INDEX` exist?* On the public demo, many visitors share one server. If uploads were saved to `index_storage/`, one visitor's documents would replace everyone's saved index. The Dockerfile sets it to `false`.

### 6.2 The LLM client: src/llm/llm_client.py

Imports: no project files. (This file is identical to the one in the Financial Analysis Agent Crew project.)

Free LLM tiers are rate-limited. This client tries each provider in order and returns the first good answer, and it mimics the OpenAI client's shape (`client.chat.completions.create`).

**`Provider`** [src/llm/llm_client.py:26-46](src/llm/llm_client.py#L26-L46). A dataclass: name, URL, which env variables may hold the key, the model variable and default model, plus `extra_body` (provider-only parameters). Its **`api_key`** property ([lines 36-42](src/llm/llm_client.py#L36-L42)) returns the first non-empty key that doesn't start with `your_`; **`model`** ([lines 44-46](src/llm/llm_client.py#L44-L46)) returns the override or the default. Properties re-read the environment each time.

**`PROVIDERS`** [src/llm/llm_client.py:49-73](src/llm/llm_client.py#L49-L73). Groq, Gemini, Mistral. Groq sends `reasoning_effort: "low"` because its `gpt-oss` model "thinks" first, and that thinking would otherwise use up the answer's token budget.

**`_Completions.create`** [src/llm/llm_client.py:80-104](src/llm/llm_client.py#L80-L104). The fallback loop:

```python
for provider, client in self._client.providers:
    try:
        response = client.chat.completions.create(
            model=provider.model,
            messages=messages,
            extra_body=provider.extra_body or None,
            **kwargs,
        )
        if response.choices and response.choices[0].message.content:
            self._client.last_provider = provider.name
            return response
        errors.append(f"{provider.name}: empty response")
    except Exception as e:
        ...
raise RuntimeError("All LLM providers failed - " + " | ".join(errors))
```

An exception **or** an empty answer moves on to the next provider; only if all fail does it raise, listing every provider's error.

**`LLMClient.__init__(order)`** [src/llm/llm_client.py:117-137](src/llm/llm_client.py#L117-L137). For each provider in `order` that has a key, it creates an `OpenAI` client pointed at that provider's URL, with `max_retries=0, timeout=30` ([line 127](src/llm/llm_client.py#L127)). *Why no retries?* The fallback **is** the retry: moving to another provider is quicker than waiting on the same busy one. No usable key at all → `ValueError`. Unlike the finance project, `RAGGenerator` passes `LLM_PROVIDERS` from `config.py` explicitly ([src/llm/generator.py:48](src/llm/generator.py#L48)), so the client's own `"groq"` default ([line 118](src/llm/llm_client.py#L118)) never matters here.

**`model`** [src/llm/llm_client.py:140-142](src/llm/llm_client.py#L140-L142), **`provider_names`** [src/llm/llm_client.py:145-146](src/llm/llm_client.py#L145-L146) and **`configured_providers(order)`** [src/llm/llm_client.py:149-155](src/llm/llm_client.py#L149-L155). The primary model name, the provider list, and a key check that doesn't create clients (used by `app.py` before anything else).

### 6.3 Loading sources: src/ingestion/

All three loaders produce LlamaIndex `Document`s with a `source_type` in the metadata (`"pdf"`, `"csv"` or `"web"`). This **common shape** is what makes the system "multi-source": after loading, the rest of the pipeline doesn't care where a Document came from. Each loader catches its own errors, prints them, and returns what it could load.

#### src/ingestion/pdf_loader.py

- **`load_pdf(file_path)`** [src/ingestion/pdf_loader.py:9-35](src/ingestion/pdf_loader.py#L9-L35): one Document **per page** with text, metadata `source, file_name, page_number, total_pages, source_type` ([lines 22-28](src/ingestion/pdf_loader.py#L22-L28)). *Why per page?* So an answer can say "Page 4".
- **`load_pdfs(file_paths)`** [src/ingestion/pdf_loader.py:38-48](src/ingestion/pdf_loader.py#L38-L48): loops over paths ending in `.pdf`.
- **`load_pdfs_from_directory(directory)`** [src/ingestion/pdf_loader.py:51-60](src/ingestion/pdf_loader.py#L51-L60): finds PDFs in a folder and its subfolders with `os.walk`. Unused.

#### src/ingestion/csv_loader.py

- **`load_csv(file_path, text_columns, metadata_columns, max_rows=500)`** [src/ingestion/csv_loader.py:9-86](src/ingestion/csv_loader.py#L9-L86).
  1. Guesses the delimiter from the first line: `|`, tab, `;`, else comma ([lines 27-38](src/ingestion/csv_loader.py#L27-L38)).
  2. Reads at most **500 rows**, skipping malformed lines ([line 40](src/ingestion/csv_loader.py#L40)).
  3. Makes **one Document per row**, with text like this, one `column: value` line per non-empty cell ([lines 55-59](src/ingestion/csv_loader.py#L55-L59)):

```text
product_name: iPhone 15 Pro
category: Smartphones
price: 999
...
```

  Writing the column name next to each value is what lets the LLM understand a row out of context. A bare "999" means nothing; "price: 999" does.

  4. Metadata gets `row_number` (1 = first data row) and, by default, **every column's value** as a string too ([lines 63-76](src/ingestion/csv_loader.py#L63-L76)).
- **`load_csvs(file_paths, **kwargs)`** [src/ingestion/csv_loader.py:89-98](src/ingestion/csv_loader.py#L89-L98) and **`load_csvs_from_directory(directory, **kwargs)`** [src/ingestion/csv_loader.py:101-110](src/ingestion/csv_loader.py#L101-L110): the multi-file and folder versions. The folder version builds the sample index.

#### src/ingestion/web_loader.py

- **`clean_text(text)`** [src/ingestion/web_loader.py:11-15](src/ingestion/web_loader.py#L11-L15): collapses whitespace.
- **`load_url(url)`** [src/ingestion/web_loader.py:18-65](src/ingestion/web_loader.py#L18-L65):
  1. Downloads the page with a browser-like `User-Agent` header ([lines 23-28](src/ingestion/web_loader.py#L23-L28)). Some sites block requests that don't look like a browser.
  2. Parses it with Beautiful Soup and **deletes** `script`, `style`, `nav`, `footer` and `header` elements ([lines 33-34](src/ingestion/web_loader.py#L33-L34)), because menus and cookie banners are noise for search.
  3. Takes the `<title>`, then the text of `<main>`, else `<article>`, else `<body>` ([lines 37-47](src/ingestion/web_loader.py#L37-L47)).
  4. Returns one Document for the whole page with `source, title, domain, source_type` ([lines 52-57](src/ingestion/web_loader.py#L52-L57)).
- **`load_urls(urls)`** [src/ingestion/web_loader.py:68-77](src/ingestion/web_loader.py#L68-L77): only accepts addresses starting with `http://` or `https://`.

[src/ingestion/\_\_init\_\_.py:1-7](src/ingestion/__init__.py#L1-L7) re-exports the three list loaders.

### 6.4 Cleaning and chunking: src/processing/

#### src/processing/processor.py

- **`clean_text(text)`** [src/processing/processor.py:8-17](src/processing/processor.py#L8-L17): squeezes repeated spaces, limits blank lines to one, trims each line. (Same name as the web loader's function, different behaviour.)
- **`enrich_metadata(document)`** [src/processing/processor.py:20-37](src/processing/processor.py#L20-L37): adds `char_count`, `word_count`, `line_count` and a `content_hint` of `"code"`, `"tabular"` or `"text"`, guessed from keywords ([lines 30-35](src/processing/processor.py#L30-L35)). Nothing reads these fields later.
- **`process_documents(documents)`** [src/processing/processor.py:40-72](src/processing/processor.py#L40-L72): for each Document, cleans its text, skips it if under 10 characters, and builds a **new** Document with a *copy* of the metadata ([lines 61-64](src/processing/processor.py#L61-L64)). The comment explains why: `Document.text` is read-only in newer LlamaIndex versions, so you can't change it in place.

#### src/processing/chunker.py

- **`chunk_documents(documents, chunk_size=1024, chunk_overlap=200)`** [src/processing/chunker.py:12-33](src/processing/chunker.py#L12-L33):

```python
splitter = SentenceSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
nodes = splitter.get_nodes_from_documents(documents)
```

`SentenceSplitter` tries to cut at sentence boundaries so chunks don't stop mid-sentence. Sizes are measured in **tokens** (word pieces), not characters. The 200-token *overlap* repeats the end of one chunk at the start of the next, so a fact that sits on a boundary isn't lost. Each node keeps its Document's metadata, so a chunk still knows its file, page or row. A CSV row is usually much shorter than 1,024 tokens, so each row stays one chunk.

- **`chunk_with_sentence_window(...)`** [src/processing/chunker.py:36-61](src/processing/chunker.py#L36-L61): an alternative where each node is one sentence plus the surrounding sentences in metadata. Unused; it's there to experiment with.

[src/processing/\_\_init\_\_.py:1-6](src/processing/__init__.py#L1-L6) re-exports `chunk_documents` and `process_documents`.

### 6.5 Embeddings and the index: src/indexing/vector_store.py

Imports: `src.config`.

**`class LocalEmbedding(BaseEmbedding)`** [src/indexing/vector_store.py:19-48](src/indexing/vector_store.py#L19-L48). An adapter: LlamaIndex knows how to *call* an embedding model through the `BaseEmbedding` interface; this class fills that interface with a sentence-transformers model.

- **`__init__`** [lines 22-26](src/indexing/vector_store.py#L22-L26): loads `all-MiniLM-L6-v2` (about 90 MB, downloaded the first time). The import is inside the method so the library only loads when needed.
- **`_get_embedding`** [lines 28-31](src/indexing/vector_store.py#L28-L31): `self._model.encode(text)` returns a NumPy array of 384 floats; `.tolist()` turns it into a Python list, which is what LlamaIndex expects.
- `_get_query_embedding`, `_get_text_embedding` and their `async` versions [lines 33-43](src/indexing/vector_store.py#L33-L43): required by the base class; they all call `_get_embedding`. Queries and documents are embedded the same way, which matters: both must live in the same "number space" to be compared.
- **`_get_text_embeddings`** [lines 45-48](src/indexing/vector_store.py#L45-L48): embeds many texts in one batch, much faster than one at a time.

**`get_embedding_model()`** [src/indexing/vector_store.py:51-55](src/indexing/vector_store.py#L51-L55). Creates a `LocalEmbedding` and also sets it as LlamaIndex's **global** default (`Settings.embed_model`). Without that, retrievers might fall back to LlamaIndex's default, which is OpenAI's paid embedding service. Each call loads the model again.

**`create_index(nodes, dimension=384, persist_dir, persist)`** [src/indexing/vector_store.py:58-95](src/indexing/vector_store.py#L58-L95).

```python
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
...
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)
```

1. An empty FAISS index for 384-number vectors. `dimension` **must** match the embedding model's output, or adding vectors fails.
2. A `StorageContext` bundles where vectors go (FAISS) with where node text and metadata go (LlamaIndex's built-in document store). FAISS only stores numbers; the text lives alongside it.
3. `VectorStoreIndex(...)` embeds every node and adds it.
4. If `persist` is true, everything is written to `index_storage/` ([lines 90-93](src/indexing/vector_store.py#L90-L93)).

**`load_index(persist_dir)`** [src/indexing/vector_store.py:98-122](src/indexing/vector_store.py#L98-L122). The reverse: load the FAISS file and the stored nodes, then rebuild the index object with the embedding model (needed to embed future questions).

**`add_documents_to_index(index, nodes, persist_dir)`** [src/indexing/vector_store.py:125-137](src/indexing/vector_store.py#L125-L137). Inserts more nodes into an existing index and saves. Exported, but the app never calls it; uploads always build a fresh index.

[src/indexing/\_\_init\_\_.py:1-5](src/indexing/__init__.py#L1-L5) re-exports the three functions.

### 6.6 Retrieval and re-ranking: src/retrieval/

#### src/retrieval/retriever.py

Imports: `src.config`.

**`class MultiRetriever`** [src/retrieval/retriever.py:11-102](src/retrieval/retriever.py#L11-L102).

- **`__init__(index, top_k)`** [lines 19-31](src/retrieval/retriever.py#L19-L31): stores the index and creates a `VectorIndexRetriever` that returns `top_k` nodes.
- **`retrieve(query, source_type)`** [lines 33-55](src/retrieval/retriever.py#L33-L55): vector search, then optionally keep only results whose `source_type` matches (e.g. only `"pdf"`). The filter runs *after* the search, so if all top results are CSV rows, asking for PDFs returns nothing.
- **`retrieve_with_metadata_filter(query, metadata_filters)`** [lines 57-82](src/retrieval/retriever.py#L57-L82): the same idea for any metadata key/value pairs. Unused.
- **`hybrid_retrieve(query, top_k)`** [lines 84-102](src/retrieval/retriever.py#L84-L102): the method the app actually uses. Despite its name, it is **plain vector search** with more results (`top_k * 2`, [line 93](src/retrieval/retriever.py#L93)), relying on re-ranking afterwards. "Hybrid" normally means vector **plus keyword** (e.g. BM25) search; there's no keyword search here ([section 8](#8-things-to-notice)).

#### src/retrieval/reranker.py

Imports: no project files.

**Optional import** [src/retrieval/reranker.py:7-11](src/retrieval/reranker.py#L7-L11). If sentence-transformers is missing, `RERANKER_AVAILABLE` is `False` and the code falls back to plain sorting.

**`class SemanticReranker`** [src/retrieval/reranker.py:14-55](src/retrieval/reranker.py#L14-L55).

- **`__init__`** [lines 17-21](src/retrieval/reranker.py#L17-L21): loads the cross-encoder `ms-marco-MiniLM-L-6-v2`, trained on Microsoft's MS MARCO search dataset to judge "does this passage answer this query?".
- **`rerank(query, results, top_k)`** [lines 23-55](src/retrieval/reranker.py#L23-L55):

```python
pairs = [(query, r.node.get_content()) for r in results]
scores = self.model.predict(pairs)
for result, score in zip(results, scores):
    result.score = float(score)
reranked = sorted(results, key=lambda x: x.score, reverse=True)
return reranked[:top_k]
```

Each (question, chunk) pair is scored, the node's `score` is **overwritten** with that score, and the list is sorted highest first. *Why re-rank at all?* The bi-encoder compares two separately computed vectors, which is fast but approximate. The cross-encoder reads both texts together, so it catches things like negation or which entity a question is about. Using the fast model to shortlist and the accurate one to choose is a standard two-stage design.

**`rerank_results(query, results, top_k, use_semantic_reranking)`** [src/retrieval/reranker.py:58-87](src/retrieval/reranker.py#L58-L87). The function the generator calls. It creates a **new** `SemanticReranker` (loading the model again) on every call ([line 80](src/retrieval/reranker.py#L80)), and if that fails, sorts by the original score instead ([lines 85-87](src/retrieval/reranker.py#L85-L87)). That fallback has a direction bug: see [section 8](#8-things-to-notice).

**`deduplicate_results(results, threshold=0.9)`** [src/retrieval/reranker.py:90-114](src/retrieval/reranker.py#L90-L114). Keeps a result only if it shares at most 90% of its words with every result already kept. *Why?* Overlapping chunks, or the same page loaded twice, would otherwise fill the LLM's context with repeats. It compares every pair, which is fine for 30 results but would be slow for thousands.

[src/retrieval/\_\_init\_\_.py:1-6](src/retrieval/__init__.py#L1-L6) re-exports `MultiRetriever` and `rerank_results`.

### 6.7 Answering: src/llm/generator.py

Imports: `src.llm.llm_client`, `src.config`, `src.retrieval.retriever`, `src.retrieval.reranker`.

**`QA_PROMPT_TEMPLATE`** [src/llm/generator.py:14-31](src/llm/generator.py#L14-L31). The instructions sent to the LLM, with two placeholders: `{context_str}` (the chunks) and `{query_str}` (the question). Key rules: answer **only** from the context (to reduce made-up facts), analyse all CSV rows for "top N" questions, and cite as `[Source: filename]`.

**`class RAGGenerator`** [src/llm/generator.py:34-169](src/llm/generator.py#L34-L169).

- **`__init__(index, top_k=15, use_reranking=True)`** [lines 37-52](src/llm/generator.py#L37-L52): creates the `LLMClient` from `LLM_PROVIDERS` and a `MultiRetriever` with `top_k * 2` = 30.
- **`_generate_response(prompt)`** [lines 54-60](src/llm/generator.py#L54-L60): sends the whole prompt as one **user** message at `temperature=0.2` (low randomness, so answers stick close to the text) and returns the reply text.
- **`_format_context(nodes)`** [lines 62-87](src/llm/generator.py#L62-L87): numbers each chunk and labels it with its source, plus a page (PDF) or row (CSV):

```text
[Source 1: sample_products.csv (Row 2)]
product_name: Samsung Galaxy S24
...
---
[Source 2: ...]
```

  The source name comes from `file_name`, else `source`, else `title` ([line 68](src/llm/generator.py#L68)). Web pages have no `file_name`, so they show their URL.
- **`_get_sources(nodes)`** [lines 89-111](src/llm/generator.py#L89-L111): the same information as a list of dicts for the UI's "Sources" box, including the score and any page, row or domain.
- **`query(question)`** [lines 113-148](src/llm/generator.py#L113-L148): the full pipeline:

```python
results = self.retriever.hybrid_retrieve(question)
results = deduplicate_results(results)
if self.use_reranking:
    results = rerank_results(question, results, top_k=self.top_k)
...
context = self._format_context(results)
prompt = QA_PROMPT_TEMPLATE.format(context_str=context, query_str=question)
response = self._generate_response(prompt)
sources = self._get_sources(results)
return response, sources
```

  Note the numbers: `MultiRetriever` holds `top_k=30`, and `hybrid_retrieve` doubles it again, so up to **60** chunks are fetched, deduplicated, and cut to **15** by the re-ranker. If nothing is found, it returns a polite "couldn't find" message without calling the LLM.
- **`query_with_filter(question, source_type)`** [lines 150-169](src/llm/generator.py#L150-L169): the same, restricted to one source type, and without deduplication. Unused by the app.

[src/llm/\_\_init\_\_.py:1-5](src/llm/__init__.py#L1-L5) re-exports `RAGGenerator`.

### 6.8 The web app: app.py

Imports: everything public from `src`.

Remember that Streamlit re-runs this file top to bottom on every click.

- **Start-up** [app.py:13-34](app.py#L13-L34): load `.env`; `st.set_page_config` (must be the first Streamlit call); import the `src` modules inside `try`, so a missing library shows a readable error on the page instead of a blank screen. `st.stop()` ends the run.
- **`check_api_key()`** [app.py:37-42](app.py#L37-L42): stops with instructions if no provider key is set.
- **`get_sample_index()`** [app.py:45-49](app.py#L45-L49): loads both CSVs in `data/`, processes, chunks and indexes them **in memory**. `@st.cache_resource` means this runs **once per server process**, and every visitor shares the result. This is fine because the sample index is read-only.
- **`load_sample_data()`** [app.py:52-56](app.py#L52-L56): points this session at the shared sample index and creates a `RAGGenerator` for it.
- **`init_session_state()`** [app.py:59-68](app.py#L59-L68): creates `index`, `generator`, `documents_loaded` and `chat_history` on the first run.
- **`load_existing_index()`** [app.py:71-81](app.py#L71-L81): loads `index_storage/` if present.
- **`process_uploaded_files(pdf_files, csv_files, urls)`** [app.py:84-120](app.py#L84-L120): Streamlit gives uploads as in-memory files, but the loaders expect a **path**, so each upload is written to a temporary file, loaded, then deleted ([lines 93-99](app.py#L93-L99)). URLs are split one per line.
- **`main()`** [app.py:123-294](app.py#L123-L294):
  - **Sidebar** [lines 133-211](app.py#L133-L211): while nothing is loaded, it shows **Try Sample Data** and, if a saved index exists, **Load Existing Index** ([lines 136-147](app.py#L136-L147)). Then two file uploaders, a URL box, and **Process & Index Documents** ([lines 175-204](app.py#L175-L204)), which runs load → process → chunk → `create_index` and replaces this session's index and generator. `st.rerun()` redraws the page in its new state.
  - **Chat** [lines 214-273](app.py#L214-L273): replays `chat_history` (Streamlit forgets the page on every re-run, so the history must be redrawn each time), then waits for `st.chat_input`. A new question is shown, answered with `st.session_state.generator.query(query)` ([line 244](app.py#L244)), and shown with an expandable list of sources. Both messages are appended to the history ([lines 263-267](app.py#L263-L267)). **Clear Chat** empties it.
  - **Welcome text** [lines 275-289](app.py#L275-L289) when no index is loaded.
  - A final `except` shows any unexpected error with its traceback ([lines 291-294](app.py#L291-L294)).
- **Entry point** [app.py:297-298](app.py#L297-L298): Streamlit runs the file as `__main__`, so `main()` is called on every re-run.

### 6.9 Running and deploying

- **[run.bat](run.bat)**: on Windows, creates `venv` if missing, activates it, installs requirements and starts Streamlit.
- **[Dockerfile](Dockerfile)**:
  - Installs **CPU-only PyTorch** first ([line 12](Dockerfile#L12)). The default PyTorch download includes GPU libraries worth gigabytes, which a free CPU Space can't use.
  - Sets `HF_HOME` (where Hugging Face models are cached) and `PERSIST_INDEX=false` ([lines 16-19](Dockerfile#L16-L19)).
  - Downloads both models **at build time** ([line 22](Dockerfile#L22)), then sets `HF_HUB_OFFLINE=1` ([line 23](Dockerfile#L23)) so the running app never tries to download them again. Cold starts are faster and don't depend on the Hub being up.
  - Runs as UID 1000 (required by Hugging Face Spaces) and starts Streamlit on port 7860 ([line 30](Dockerfile#L30)).
- **[.github/workflows/deploy-hf.yml](.github/workflows/deploy-hf.yml)**: pushes `main` to the Hugging Face Space on every push ([line 29](.github/workflows/deploy-hf.yml#L29)), and every 12 hours ([line 9](.github/workflows/deploy-hf.yml#L9)) pings the demo so the free Space stays awake.
- **[README.md:1-10](README.md#L1-L10)**: the block between `---` lines configures the Space (Docker SDK, port 7860).

---

## 7. One question traced from start to finish

You open the app, click **Try Sample Data**, and ask **"Which smartphones cost less than $900?"**. Assume `.env` has a Gemini key.

**Part A: building the sample index**

**1. First page load.** `.env` is loaded, `src` is imported, the key check passes, and session state is created: [app.py:13](app.py#L13), [app.py:125-126](app.py#L125-L126).

**2. Click "Try Sample Data".** On that re-run the button returns `True`, so `load_sample_data()` runs: [app.py:137-139](app.py#L137-L139).

**3. Load the CSVs.** `get_sample_index()` (first call on this server) walks `data/` and loads both files: [app.py:48](app.py#L48) → [src/ingestion/csv_loader.py:101-110](src/ingestion/csv_loader.py#L101-L110).
Each file has a comma-separated header, so the delimiter is `,`. That gives 8 policy rows + 10 product rows = **18 Documents**, e.g. `product_name: Samsung Galaxy S24\ncategory: Smartphones\nprice: 899\n...` with metadata `file_name="sample_products.csv", row_number=2`: [src/ingestion/csv_loader.py:52-79](src/ingestion/csv_loader.py#L52-L79).

**4. Process and chunk.** Text is cleaned and statistics are added; every row is short, so it stays one node. **18 nodes**: [src/processing/processor.py:40-72](src/processing/processor.py#L40-L72), [src/processing/chunker.py:25-30](src/processing/chunker.py#L25-L30).

**5. Index.** The MiniLM model is loaded and each node is turned into a 384-number vector and added to a new `IndexFlatL2`, in memory only (`persist=False`): [app.py:49](app.py#L49) → [src/indexing/vector_store.py:72-87](src/indexing/vector_store.py#L72-L87).

**6. Generator.** A `RAGGenerator` is created; its `LLMClient` gets one OpenAI-compatible client for Gemini: [app.py:55](app.py#L55) → [src/llm/generator.py:47-52](src/llm/generator.py#L47-L52). `st.rerun()` redraws the page, now showing the chat.

**Part B: answering the question**

**7. You type the question.** `st.chat_input` returns the text; it's added to the history and shown: [app.py:233-238](app.py#L233-L238).

**8. Retrieve.** `generator.query()` → `hybrid_retrieve()` asks for the 60 nearest vectors; there are only 18, so FAISS returns all 18 with their L2 distances: [app.py:244](app.py#L244) → [src/llm/generator.py:124](src/llm/generator.py#L124) → [src/retrieval/retriever.py:93-101](src/retrieval/retriever.py#L93-L101).

**9. Deduplicate.** No two rows share 90% of their words, so all 18 stay: [src/retrieval/reranker.py:90-114](src/retrieval/reranker.py#L90-L114).

**10. Re-rank.** The cross-encoder scores 18 (question, row) pairs; smartphone rows score highest. The top **15** are kept: [src/llm/generator.py:130-131](src/llm/generator.py#L130-L131) → [src/retrieval/reranker.py:78-81](src/retrieval/reranker.py#L78-L81) → [src/retrieval/reranker.py:43-55](src/retrieval/reranker.py#L43-L55).

**11. Build the prompt.** The 15 rows are numbered and labelled (`[Source 1: sample_products.csv (Row 2)]`…) and inserted into the template with the question: [src/llm/generator.py:139-142](src/llm/generator.py#L139-L142) → [src/llm/generator.py:62-87](src/llm/generator.py#L62-L87).

**12. Ask the LLM.** One user message goes to Gemini at temperature 0.2. If Gemini were rate-limited, the same message would go to Mistral, then Groq: [src/llm/generator.py:54-60](src/llm/generator.py#L54-L60) → [src/llm/llm_client.py:88-104](src/llm/llm_client.py#L88-L104).
The LLM reads the price lines and answers, for example, that the Samsung Galaxy S24 at $899 is the only smartphone under $900 (the iPhone 15 Pro is $999), citing `sample_products.csv`.

**13. Show it.** The answer is printed, the 15 sources are listed with row numbers, and both are stored in `chat_history`: [app.py:246-267](app.py#L246-L267).

Why did a "less than $900" question work? Because the sample data is so small that every product row reached the LLM. With a 500-row CSV, only 15 rows would, so the LLM could miss cheaper products it never saw. Questions that need **every** row (counting, filtering, sorting) are a known weak spot of RAG; see [exercise 10](#9-exercises).

---

## 8. Things to notice

Real issues found while reading the code (and, for the FAISS scores, the LlamaIndex source). Each one is a small lesson.

### Bugs

1. **Uploaded files are cited by their temporary name.** `process_uploaded_files` saves each upload to a temp file like `/tmp/tmpk3j9x1.pdf` ([app.py:93-97](app.py#L93-L97)), and the loaders take `file_name` from that path ([src/ingestion/pdf_loader.py:15](src/ingestion/pdf_loader.py#L15), [src/ingestion/csv_loader.py:41](src/ingestion/csv_loader.py#L41)). So answers cite `[Source: tmpk3j9x1.pdf]` instead of `annual_report.pdf`. Fix: after loading, set `doc.metadata["file_name"] = pdf_file.name` ([exercise 2](#9-exercises)).

2. **The fallback ranking is upside down.** LlamaIndex's FAISS store reports raw **L2 distances** as `score` (smaller = closer). The cross-encoder path replaces them with relevance scores (bigger = better), so the main path is fine. But if the cross-encoder fails to load, `rerank_results` sorts by the original score with `reverse=True` ([src/retrieval/reranker.py:86](src/retrieval/reranker.py#L86)), putting the **least** similar chunks first. The `score` shown in sources therefore also means different things depending on the path.

3. **Web text cleaning loses all line breaks.** In the web loader's `clean_text`, the first line replaces *every* whitespace run (including newlines) with a space ([src/ingestion/web_loader.py:13](src/ingestion/web_loader.py#L13)), so the second line, which looks for blank lines, can never match ([line 14](src/ingestion/web_loader.py#L14)). Every web page becomes one long line, which makes sentence splitting less accurate.

4. **"Hybrid" retrieval is vector-only.** `hybrid_retrieve` ([src/retrieval/retriever.py:84-102](src/retrieval/retriever.py#L84-L102)) and the README both say hybrid, but there is no keyword (BM25) search. Exact terms like product codes or names can be missed by vector search alone ([exercise 9](#9-exercises)).

### Design quirks

5. **Models are reloaded again and again.** A new `CrossEncoder` is created on **every question** ([src/retrieval/reranker.py:80](src/retrieval/reranker.py#L80)), and a new `SentenceTransformer` every time an index is created or loaded ([src/indexing/vector_store.py:53](src/indexing/vector_store.py#L53)). Loading takes about a second each time. Caching them once (for example with `functools.lru_cache` or `st.cache_resource`) would make answers noticeably faster.

6. **Each upload replaces the index; it doesn't add to it.** "Process & Index Documents" always calls `create_index` ([app.py:195](app.py#L195)), so earlier uploads (and the sample data) are dropped from the session, and with `PERSIST_INDEX=true`, `index_storage/` is overwritten. `add_documents_to_index` exists but isn't used. The chat history is also kept, even though it now refers to a different index.

7. **CSV metadata is very large.** Every column value is copied into metadata ([src/ingestion/csv_loader.py:71-76](src/ingestion/csv_loader.py#L71-L76)), plus the path and statistics added by `enrich_metadata`. By default LlamaIndex includes metadata when it embeds a node and counts it against the chunk size. So each row's values are embedded twice, and a CSV with a long text column can make `SentenceSplitter` raise *"Metadata length (…) is longer than chunk size"* (an error confirmed in LlamaIndex's source). Passing `metadata_columns=[]` would avoid both.

8. **CSV limits.** Only the first **500 rows** are read ([src/ingestion/csv_loader.py:13](src/ingestion/csv_loader.py#L13)), silently. The delimiter guess looks at the first line only, so a comma CSV whose header contains `;` or `|` will be split wrongly ([src/ingestion/csv_loader.py:31-38](src/ingestion/csv_loader.py#L31-L38)). Malformed lines are skipped (`on_bad_lines='skip'`). A non-UTF-8 file fails to load and returns no rows.

9. **Top-k numbers are confusing.** `RAGGenerator` creates `MultiRetriever(top_k=30)` ([src/llm/generator.py:52](src/llm/generator.py#L52)), whose own `vector_retriever` is only used by the unused filter methods, while `hybrid_retrieve` builds a **new** retriever with 60 ([src/retrieval/retriever.py:93-99](src/retrieval/retriever.py#L93-L99)).

10. **Source filtering happens after the search** ([src/retrieval/retriever.py:49-53](src/retrieval/retriever.py#L49-L53)), so it can return nothing even when matching documents exist. LlamaIndex's FAISS store doesn't support metadata filters itself, which is probably why it was done this way.

11. **The prompt mentions a "Rank" column** ([src/llm/generator.py:24](src/llm/generator.py#L24)) that neither sample CSV has. It reads like an instruction tuned for an earlier dataset.

12. **Fetching any URL from the server.** On a public demo, `load_url` will download any address a visitor types ([src/ingestion/web_loader.py:27](src/ingestion/web_loader.py#L27)), including internal addresses that the server can reach but the visitor can't. In security this is called SSRF; a real deployment should block private IP ranges.

13. **`content_hint` guesses are rough.** `'row' in text` also matches "grow" and "throw", and `'data'` is in almost everything ([src/processing/processor.py:30-35](src/processing/processor.py#L30-L35)). Since nothing reads `content_hint`, it does no harm, but it adds noise to the metadata.

### Unused code

- `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `SIMILARITY_THRESHOLD` in [src/config.py](src/config.py#L14-L22). Changing them does nothing (see [exercise 1](#9-exercises)).
- `chunk_with_sentence_window` ([src/processing/chunker.py:36-61](src/processing/chunker.py#L36-L61)), `retrieve_with_metadata_filter` ([src/retrieval/retriever.py:57-82](src/retrieval/retriever.py#L57-L82)), `query_with_filter` ([src/llm/generator.py:150-169](src/llm/generator.py#L150-L169)), `add_documents_to_index` ([src/indexing/vector_store.py:125-137](src/indexing/vector_store.py#L125-L137)), `load_pdfs_from_directory` ([src/ingestion/pdf_loader.py:51-60](src/ingestion/pdf_loader.py#L51-L60)).
- `LLMClient.last_provider` (set, never read) and `provider_names`; `RAGGenerator.model_name`.
- The `char_count`, `word_count`, `line_count` and `content_hint` metadata fields.
- `uploads/` and `documents/` in `.gitignore`/`.dockerignore`. No code writes there.

### Setup gotchas

- **First run downloads models** (about 90 MB each for the embedder and cross-encoder) from Hugging Face. Without internet the first index build fails.
- **PyTorch is big.** `pip install -r requirements.txt` pulls in the full PyTorch. On a laptop without a GPU, the Dockerfile's trick of installing the CPU wheel first saves gigabytes.
- **`llama-index` is a large meta-package.** Only `llama-index-core` and the FAISS store are actually used; installing those two would be much smaller.
- **Inline comments in `.env`**: `GEMINI_API_KEY=    # https://...` ([.env.example:6](.env.example#L6)) is read as empty. Put your key straight after `=`.
- **Scanned PDFs** (images of text) load as zero pages, because `extract_text()` finds no text layer and there's no OCR.
- **Run from the project root** (`streamlit run app.py`), so `from src...` imports work.

---

## 9. Exercises

From easy to advanced. Each says which file to change.

1. **Make the config real.** Use `CHUNK_SIZE`/`CHUNK_OVERLAP` as `chunk_documents`' defaults and `EMBEDDING_MODEL` in `get_embedding_model`. *Files:* [src/processing/chunker.py:12-16](src/processing/chunker.py#L12-L16), [src/indexing/vector_store.py:51-55](src/indexing/vector_store.py#L51-L55), [src/config.py:14-18](src/config.py#L14-L18). Then try a chunk size of 256 on a PDF and compare answers.

2. **Cite real file names.** After loading each uploaded file, overwrite `file_name` (and `source`) in every Document's metadata with the uploaded file's real name. *File:* [app.py:89-111](app.py#L89-L111).

3. **Show the provider.** Under each answer, show which LLM answered (`generator.client.last_provider`). *Files:* [app.py:244-256](app.py#L244-L256).

4. **Fix the fallback sort.** In `rerank_results`, sort FAISS distances ascending (smaller is better) when the cross-encoder isn't used. *File:* [src/retrieval/reranker.py:85-87](src/retrieval/reranker.py#L85-L87). Test with `use_semantic_reranking=False`.

5. **Load models once.** Cache the cross-encoder and the embedding model so they are created once per process. *Files:* [src/retrieval/reranker.py:78-81](src/retrieval/reranker.py#L78-L81), [src/indexing/vector_store.py:51-55](src/indexing/vector_store.py#L51-L55). Time a question before and after.

6. **Keep line breaks from web pages.** Fix the web loader's `clean_text` so paragraphs survive (collapse spaces and tabs, but keep newlines). *File:* [src/ingestion/web_loader.py:11-15](src/ingestion/web_loader.py#L11-L15). Bonus: use `get_text(separator="\n")` in [src/ingestion/web_loader.py:45](src/ingestion/web_loader.py#L45).

7. **Add documents instead of replacing.** If an index already exists in the session, call `add_documents_to_index` (or `index.insert_nodes`) instead of `create_index`. *Files:* [app.py:193-197](app.py#L193-L197), [src/indexing/vector_store.py:125-137](src/indexing/vector_store.py#L125-L137).

8. **Source filter in the UI.** Add a "Search in: All / PDF / CSV / Web" selector and use `query_with_filter`. Then fix the filter so it applies *before* cutting to top-k (retrieve more, filter, then re-rank). *Files:* [app.py:232-244](app.py#L232-L244), [src/llm/generator.py:150-169](src/llm/generator.py#L150-L169), [src/retrieval/retriever.py:33-55](src/retrieval/retriever.py#L33-L55).

9. **Real hybrid retrieval.** Add BM25 keyword search (e.g. the `rank-bm25` package over the nodes' text), merge its results with the vector results, then let the cross-encoder pick the best. *File:* [src/retrieval/retriever.py:84-102](src/retrieval/retriever.py#L84-L102). Test with an exact product name.

10. **Answer whole-table questions.** For CSV questions like "how many products cost under $500?", RAG sees at most 15 rows. Add a path that loads the CSV into pandas and asks the LLM to write a pandas filter expression, or compute simple counts directly. Then compare its answers with the RAG answers on a 200-row CSV. *Files:* new module in `src/`, wired into [src/llm/generator.py:113-148](src/llm/generator.py#L113-L148).

---

## 10. Glossary

**Bi-encoder.** A model that turns one text into a vector on its own. Fast; used to embed all chunks ahead of time.

**Chunk / node.** A small piece of a document that is embedded and retrieved on its own. LlamaIndex calls it a node.

**Citation.** A note saying which source a fact came from, e.g. `[Source: sample_products.csv (Row 2)]`.

**Cross-encoder.** A model that reads a question and a passage together and outputs one relevance score. Accurate but slow, so only used to re-rank a shortlist.

**Deduplication.** Removing near-identical results so the LLM doesn't see the same text twice.

**Document (LlamaIndex).** Source text plus a metadata dict, before chunking.

**Embedding / vector.** A list of numbers representing a text's meaning; similar texts give nearby vectors. Here, 384 numbers.

**FAISS.** A library for fast nearest-vector search. `IndexFlatL2` compares the query to every stored vector by L2 distance.

**Fallback.** A second option used automatically when the first fails, here the next LLM provider.

**Hybrid search.** Combining vector (meaning) search with keyword (exact word) search. Not yet implemented here.

**L2 distance.** Straight-line distance between two vectors; smaller means more similar.

**LLM (Large Language Model).** An AI model that generates text, like Gemini, Mistral or `gpt-oss` on Groq.

**Metadata.** Extra facts about a document or chunk (file name, page, row, source type) stored alongside its text.

**OpenAI-compatible API.** A service that accepts the same request format as OpenAI's chat API.

**Persist.** Save to disk so it survives a restart. Here, the index in `index_storage/`.

**Prompt template.** Fixed instructions with placeholders (`{context_str}`, `{query_str}`) filled in for each question.

**RAG (Retrieval-Augmented Generation).** Retrieve relevant text first, then have the LLM answer using it.

**Rate limit.** A cap on requests per time period; exceeding it returns HTTP 429.

**Re-ranking.** Scoring retrieved results again with a better model and re-ordering them.

**Retriever.** The component that, given a question, returns the most relevant chunks.

**Session state (Streamlit).** A per-visitor dict that survives the script re-running on each interaction.

**Temperature.** An LLM setting for randomness: low is consistent and literal, high is creative.

**Token.** A word piece that models read and write; chunk sizes here are measured in tokens.

**Top-k.** "The best k results", e.g. the 15 chunks sent to the LLM.

**Vector store / vector index.** Where embeddings are stored and searched. Here, FAISS through LlamaIndex.
