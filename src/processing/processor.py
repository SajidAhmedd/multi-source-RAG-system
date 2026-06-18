"""Document processing pipeline for cleaning and enriching documents."""

import re
from typing import List
from llama_index.core import Document


def clean_text(text: str) -> str:
    """Clean document text by removing extra whitespace and artifacts."""
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def enrich_metadata(document: Document) -> Document:
    """Add computed metadata to document."""
    text = document.text

    # Add text statistics
    document.metadata["char_count"] = len(text)
    document.metadata["word_count"] = len(text.split())
    document.metadata["line_count"] = len(text.split('\n'))

    # Detect content type hints
    if any(keyword in text.lower() for keyword in ['def ', 'class ', 'import ', 'function']):
        document.metadata["content_hint"] = "code"
    elif any(keyword in text.lower() for keyword in ['table', 'column', 'row', 'data']):
        document.metadata["content_hint"] = "tabular"
    else:
        document.metadata["content_hint"] = "text"

    return document


def process_documents(documents: List[Document]) -> List[Document]:
    """
    Process documents through cleaning and enrichment pipeline.

    Args:
        documents: Raw documents from ingestion

    Returns:
        Processed documents ready for indexing
    """
    processed = []

    for doc in documents:
        # Clean text
        cleaned_text = clean_text(doc.text)

        # Skip empty documents
        if not cleaned_text or len(cleaned_text) < 10:
            continue

        # Create new document with cleaned text (Document.text is read-only in newer versions)
        new_doc = Document(
            text=cleaned_text,
            metadata=doc.metadata.copy()
        )

        # Enrich metadata
        new_doc = enrich_metadata(new_doc)
        processed.append(new_doc)

    print(f"Processed {len(processed)} documents (filtered {len(documents) - len(processed)} empty)")

    return processed
