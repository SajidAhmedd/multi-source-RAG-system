"""PDF document loader for the RAG system."""

import os
from typing import List
from llama_index.core import Document
from pypdf import PdfReader


def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF file and convert to LlamaIndex Documents."""
    documents = []

    try:
        reader = PdfReader(file_path)
        file_name = os.path.basename(file_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                doc = Document(
                    text=text,
                    metadata={
                        "source": file_path,
                        "file_name": file_name,
                        "page_number": page_num + 1,
                        "total_pages": len(reader.pages),
                        "source_type": "pdf"
                    }
                )
                documents.append(doc)

    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")

    return documents


def load_pdfs(file_paths: List[str]) -> List[Document]:
    """Load multiple PDF files."""
    all_documents = []

    for file_path in file_paths:
        if file_path.lower().endswith('.pdf'):
            docs = load_pdf(file_path)
            all_documents.extend(docs)
            print(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}")

    return all_documents


def load_pdfs_from_directory(directory: str) -> List[Document]:
    """Load all PDFs from a directory."""
    pdf_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    return load_pdfs(pdf_files)
