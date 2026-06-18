"""
Advanced Multi-Source RAG for Enterprise Knowledge Base
Streamlit Web Application
"""

import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Enterprise Knowledge Base RAG",
    page_icon="🔍",
    layout="wide"
)

# Try imports with error handling
try:
    from src.ingestion import load_pdfs, load_csvs, load_urls
    from src.processing import process_documents, chunk_documents
    from src.indexing import create_index, load_index
    from src.llm import RAGGenerator
    from src.config import INDEX_DIR, GOOGLE_API_KEY
except Exception as e:
    st.error(f"Import Error: {e}")
    st.code(str(e))
    st.stop()


def check_api_key():
    """Check if API key is configured."""
    if not GOOGLE_API_KEY:
        st.error("⚠️ Please set your GOOGLE_API_KEY in the .env file")
        st.info("Get your free API key at: https://aistudio.google.com/apikey")
        st.stop()


def init_session_state():
    """Initialize session state variables."""
    if "index" not in st.session_state:
        st.session_state.index = None
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def load_existing_index():
    """Try to load an existing index."""
    try:
        if os.path.exists(INDEX_DIR):
            st.session_state.index = load_index()
            st.session_state.generator = RAGGenerator(st.session_state.index)
            st.session_state.documents_loaded = True
            return True
    except Exception as e:
        st.warning(f"Could not load existing index: {e}")
    return False


def process_uploaded_files(pdf_files, csv_files, urls):
    """Process uploaded files and URLs."""
    all_documents = []

    # Process PDFs
    if pdf_files:
        with st.spinner("Processing PDF files..."):
            for pdf_file in pdf_files:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name

                docs = load_pdfs([tmp_path])
                all_documents.extend(docs)
                os.unlink(tmp_path)

    # Process CSVs
    if csv_files:
        with st.spinner("Processing CSV files..."):
            for csv_file in csv_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(csv_file.read())
                    tmp_path = tmp.name

                docs = load_csvs([tmp_path])
                all_documents.extend(docs)
                os.unlink(tmp_path)

    # Process URLs
    if urls:
        with st.spinner("Fetching web content..."):
            url_list = [u.strip() for u in urls.split("\n") if u.strip()]
            docs = load_urls(url_list)
            all_documents.extend(docs)

    return all_documents


def main():
    try:
        check_api_key()
        init_session_state()

        # Header
        st.title("🔍 Enterprise Knowledge Base RAG")
        st.markdown("*Ask questions across PDFs, CSVs, and websites with AI-powered answers*")

        # Sidebar for document upload
        with st.sidebar:
            st.header("📁 Data Sources")

            # Try to load existing index
            if not st.session_state.documents_loaded:
                if st.button("Load Existing Index"):
                    if load_existing_index():
                        st.success("Index loaded successfully!")
                        st.rerun()

            st.divider()

            # File uploaders
            st.subheader("Upload Documents")

            pdf_files = st.file_uploader(
                "PDF Files",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader"
            )

            csv_files = st.file_uploader(
                "CSV Files",
                type=["csv"],
                accept_multiple_files=True,
                key="csv_uploader"
            )

            urls = st.text_area(
                "Website URLs (one per line)",
                placeholder="",
                key="url_input"
            )

            # Process button
            if st.button("Process & Index Documents", type="primary"):
                if not pdf_files and not csv_files and not urls:
                    st.warning("Please upload at least one document or enter a URL")
                else:
                    try:
                        with st.spinner("Processing documents..."):
                            # Load documents
                            documents = process_uploaded_files(pdf_files, csv_files, urls)

                            if documents:
                                # Process documents
                                documents = process_documents(documents)
                                st.info(f"Processed {len(documents)} documents")

                                # Chunk documents
                                nodes = chunk_documents(documents)
                                st.info(f"Created {len(nodes)} chunks")

                                # Create index
                                with st.spinner("Creating vector index..."):
                                    st.session_state.index = create_index(nodes)
                                    st.session_state.generator = RAGGenerator(st.session_state.index)
                                    st.session_state.documents_loaded = True

                                st.success("Documents indexed successfully!")
                                st.rerun()
                            else:
                                st.error("No documents could be loaded")
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")

            # Index status
            st.divider()
            if st.session_state.documents_loaded:
                st.success("✅ Index Ready")
            else:
                st.info("📝 Upload documents to get started")

        # Main chat area
        if st.session_state.documents_loaded and st.session_state.generator:
            # Chat interface
            st.subheader("💬 Ask Questions")

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("📚 Sources"):
                            for src in message["sources"]:
                                source_text = f"**{src['source']}**"
                                if src.get("page"):
                                    source_text += f" (Page {src['page']})"
                                if src.get("row"):
                                    source_text += f" (Row {src['row']})"
                                st.markdown(f"- {source_text}")

            # Query input
            if query := st.chat_input("Ask a question about your documents..."):
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": query})

                with st.chat_message("user"):
                    st.markdown(query)

                # Generate response
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Thinking..."):
                            answer, sources = st.session_state.generator.query(query)

                        st.markdown(answer)

                        if sources:
                            with st.expander("📚 Sources"):
                                for src in sources:
                                    source_text = f"**{src['source']}**"
                                    if src.get("page"):
                                        source_text += f" (Page {src['page']})"
                                    if src.get("row"):
                                        source_text += f" (Row {src['row']})"
                                    st.markdown(f"- {source_text}")
                    except Exception as e:
                        answer = f"Error generating response: {e}"
                        sources = []
                        st.error(answer)

                # Add assistant message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            # Clear chat button
            if st.session_state.chat_history:
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()

        else:
            # Welcome message
            st.info("👈 Upload documents in the sidebar to get started!")

            st.markdown("""
            ### How to use:
            1. **Upload documents** - PDFs, CSVs, or enter website URLs
            2. **Click "Process & Index Documents"** - This creates a searchable index
            3. **Ask questions** - The AI will answer using your documents with citations

            ### Supported sources:
            - 📄 **PDF files** - Reports, papers, documentation
            - 📊 **CSV files** - Structured data, spreadsheets
            - 🌐 **Websites** - Any public webpage
            """)

    except Exception as e:
        st.error(f"Application Error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
