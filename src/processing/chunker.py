"""Document chunking utilities for the RAG system."""

from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
)
from llama_index.core.schema import TextNode


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 200
) -> List[TextNode]:
    """
    Chunk documents using sentence-aware splitting.

    Args:
        documents: List of LlamaIndex Documents
        chunk_size: Target size for each chunk
        chunk_overlap: Overlap between chunks for context continuity
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks from {len(documents)} documents")

    return nodes


def chunk_with_sentence_window(
    documents: List[Document],
    window_size: int = 3,
    window_metadata_key: str = "window",
    original_text_metadata_key: str = "original_text"
) -> List[TextNode]:
    """
    Chunk documents with sentence window context.
    Each chunk stores surrounding sentences for better context during retrieval.

    Args:
        documents: List of LlamaIndex Documents
        window_size: Number of sentences on each side to include
        window_metadata_key: Key to store window text in metadata
        original_text_metadata_key: Key to store original sentence
    """
    parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key=window_metadata_key,
        original_text_metadata_key=original_text_metadata_key,
    )

    nodes = parser.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} sentence-window chunks from {len(documents)} documents")

    return nodes
