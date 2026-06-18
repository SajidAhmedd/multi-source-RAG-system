"""CSV/structured data loader for the RAG system."""

import os
from typing import List, Optional
import pandas as pd
from llama_index.core import Document


def load_csv(
    file_path: str,
    text_columns: Optional[List[str]] = None,
    metadata_columns: Optional[List[str]] = None,
    max_rows: Optional[int] = 500  # Limit rows for faster testing
) -> List[Document]:
    """
    Load a CSV file and convert rows to LlamaIndex Documents.

    Args:
        file_path: Path to the CSV file
        text_columns: Columns to combine as document text. If None, uses all columns.
        metadata_columns: Columns to include as metadata. If None, includes all.
    """
    documents = []

    try:
        # Try to detect delimiter
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()

        # Detect delimiter
        if '|' in first_line:
            delimiter = '|'
        elif '\t' in first_line:
            delimiter = '\t'
        elif ';' in first_line:
            delimiter = ';'
        else:
            delimiter = ','

        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', on_bad_lines='skip', nrows=max_rows)
        file_name = os.path.basename(file_path)
        print(f"Loading {len(df)} rows (max_rows={max_rows})")

        # If no text columns specified, use all columns
        if text_columns is None:
            text_columns = df.columns.tolist()

        # If no metadata columns specified, use all columns
        if metadata_columns is None:
            metadata_columns = df.columns.tolist()

        for idx, row in df.iterrows():
            # Combine text columns into document text
            text_parts = []
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")

            text = "\n".join(text_parts)

            if text.strip():
                # Build metadata
                metadata = {
                    "source": file_path,
                    "file_name": file_name,
                    "row_number": idx + 1,
                    "total_rows": len(df),
                    "source_type": "csv"
                }

                # Add specified metadata columns
                for col in metadata_columns:
                    if col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            metadata[col] = str(value)

                doc = Document(text=text, metadata=metadata)
                documents.append(doc)

        print(f"Loaded {len(documents)} rows from {file_name}")

    except Exception as e:
        print(f"Error loading CSV {file_path}: {e}")

    return documents


def load_csvs(file_paths: List[str], **kwargs) -> List[Document]:
    """Load multiple CSV files."""
    all_documents = []

    for file_path in file_paths:
        if file_path.lower().endswith('.csv'):
            docs = load_csv(file_path, **kwargs)
            all_documents.extend(docs)

    return all_documents


def load_csvs_from_directory(directory: str, **kwargs) -> List[Document]:
    """Load all CSVs from a directory."""
    csv_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    return load_csvs(csv_files, **kwargs)
