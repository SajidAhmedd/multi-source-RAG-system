"""Web/URL content loader for the RAG system."""

import re
from typing import List
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document


def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def load_url(url: str) -> List[Document]:
    """Load content from a single URL and convert to LlamaIndex Document."""
    documents = []

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else urlparse(url).netloc

        # Extract main content
        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('body')

        if main_content:
            text = clean_text(main_content.get_text())
        else:
            text = clean_text(soup.get_text())

        if text:
            doc = Document(
                text=text,
                metadata={
                    "source": url,
                    "title": title_text,
                    "domain": urlparse(url).netloc,
                    "source_type": "web"
                }
            )
            documents.append(doc)
            print(f"Loaded content from {url[:50]}...")

    except Exception as e:
        print(f"Error loading URL {url}: {e}")

    return documents


def load_urls(urls: List[str]) -> List[Document]:
    """Load content from multiple URLs."""
    all_documents = []

    for url in urls:
        if url.startswith(('http://', 'https://')):
            docs = load_url(url)
            all_documents.extend(docs)

    return all_documents
