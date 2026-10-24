"""FAISS vector store implementation for the RAG system."""

import os
from typing import List, Optional
import faiss
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from ..config import INDEX_DIR


class LocalEmbedding(BaseEmbedding):
    """Local embedding using HuggingFace sentence-transformers. Fast and free!"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding - much faster!"""
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()


def get_embedding_model() -> LocalEmbedding:
    """Get the local embedding model."""
    embed_model = LocalEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    return embed_model


def create_index(
    nodes: List[TextNode],
    dimension: int = 384,  # all-MiniLM-L6-v2 produces 384-dim vectors
    persist_dir: Optional[str] = None,
    persist: bool = True
) -> VectorStoreIndex:
    """
    Create a new FAISS vector index from nodes.

    With persist=False the index lives only in memory.
    """
    persist_dir = persist_dir or INDEX_DIR

    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Get embedding model
    embed_model = get_embedding_model()

    # Create the index
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    # Persist to disk
    if persist:
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"Index saved to {persist_dir}")

    return index


def load_index(persist_dir: Optional[str] = None) -> VectorStoreIndex:
    """Load an existing index from disk."""
    persist_dir = persist_dir or INDEX_DIR

    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No index found at {persist_dir}")

    # Load FAISS index
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=persist_dir
    )

    # Get embedding model
    embed_model = get_embedding_model()

    # Load the index
    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model
    )

    print(f"Index loaded from {persist_dir}")
    return index


def add_documents_to_index(
    index: VectorStoreIndex,
    nodes: List[TextNode],
    persist_dir: Optional[str] = None
) -> VectorStoreIndex:
    """Add new documents to an existing index."""
    persist_dir = persist_dir or INDEX_DIR

    index.insert_nodes(nodes)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Added {len(nodes)} nodes to index")

    return index
