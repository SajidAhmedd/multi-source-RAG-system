"""Data ingestion modules for loading documents from multiple sources."""

from .pdf_loader import load_pdfs
from .csv_loader import load_csvs
from .web_loader import load_urls

__all__ = ["load_pdfs", "load_csvs", "load_urls"]
