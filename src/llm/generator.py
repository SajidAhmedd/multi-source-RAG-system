"""Response generation with LLM and source citations."""

from typing import List, Tuple
from .llm_client import LLMClient
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from ..config import LLM_PROVIDERS, TOP_K_RESULTS
from ..retrieval.retriever import MultiRetriever
from ..retrieval.reranker import rerank_results, deduplicate_results


# Custom prompt template with citation instructions
QA_PROMPT_TEMPLATE = """You are an intelligent assistant that answers questions based on the provided context.

Context information from multiple sources:
---------------------
{context_str}
---------------------

Instructions:
1. Answer the question based ONLY on the context provided above
2. If the context contains structured data (like CSV with columns), analyze ALL rows provided and sort/filter as needed
3. For "top N" questions, look at ranking columns (like "Rank") and return items with the lowest rank numbers
4. Include source citations in your answer using [Source: filename] format
5. Be concise but thorough - list items clearly
6. If information comes from multiple sources, mention all relevant sources

Question: {query_str}

Answer with citations:"""


class RAGGenerator:
    """RAG-based response generator with source citations."""

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = TOP_K_RESULTS,
        use_reranking: bool = True
    ):
        self.index = index
        self.top_k = top_k
        self.use_reranking = use_reranking

        # LLM client - tries each provider in LLM_PROVIDERS in order
        self.client = LLMClient(LLM_PROVIDERS)
        self.model_name = self.client.model

        # Initialize retriever
        self.retriever = MultiRetriever(index, top_k=top_k * 2)

    def _generate_response(self, prompt: str) -> str:
        """Generate response, falling back across providers if one fails."""
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _format_context(self, nodes: List[NodeWithScore]) -> str:
        """Format retrieved nodes into context string with source info."""
        context_parts = []

        for i, node in enumerate(nodes, 1):
            metadata = node.node.metadata
            source = metadata.get("file_name") or metadata.get("source") or metadata.get("title", "Unknown")
            source_type = metadata.get("source_type", "document")

            # Add source-specific metadata
            extra_info = ""
            if source_type == "pdf":
                page = metadata.get("page_number", "")
                if page:
                    extra_info = f" (Page {page})"
            elif source_type == "csv":
                row = metadata.get("row_number", "")
                if row:
                    extra_info = f" (Row {row})"

            content = node.node.get_content()
            context_parts.append(
                f"[Source {i}: {source}{extra_info}]\n{content}\n"
            )

        return "\n---\n".join(context_parts)

    def _get_sources(self, nodes: List[NodeWithScore]) -> List[dict]:
        """Extract source information from nodes."""
        sources = []

        for node in nodes:
            metadata = node.node.metadata
            source_info = {
                "source": metadata.get("file_name") or metadata.get("source") or metadata.get("title", "Unknown"),
                "source_type": metadata.get("source_type", "document"),
                "score": node.score,
            }

            # Add source-specific details
            if metadata.get("page_number"):
                source_info["page"] = metadata["page_number"]
            if metadata.get("row_number"):
                source_info["row"] = metadata["row_number"]
            if metadata.get("domain"):
                source_info["domain"] = metadata["domain"]

            sources.append(source_info)

        return sources

    def query(self, question: str) -> Tuple[str, List[dict]]:
        """
        Answer a question using RAG.

        Args:
            question: User's question

        Returns:
            Tuple of (answer, list of sources)
        """
        # Retrieve relevant context
        results = self.retriever.hybrid_retrieve(question)

        # Deduplicate
        results = deduplicate_results(results)

        # Re-rank if enabled
        if self.use_reranking:
            results = rerank_results(question, results, top_k=self.top_k)
        else:
            results = results[:self.top_k]

        if not results:
            return "I couldn't find any relevant information to answer your question.", []

        # Format context
        context = self._format_context(results)

        # Generate response using prompt template
        prompt = QA_PROMPT_TEMPLATE.format(context_str=context, query_str=question)
        response = self._generate_response(prompt)

        # Extract sources
        sources = self._get_sources(results)

        return response, sources

    def query_with_filter(
        self,
        question: str,
        source_type: str = None
    ) -> Tuple[str, List[dict]]:
        """Query with optional source type filter."""
        results = self.retriever.retrieve(question, source_type=source_type)

        if self.use_reranking:
            results = rerank_results(question, results, top_k=self.top_k)

        if not results:
            return f"I couldn't find any relevant information from {source_type or 'any'} sources.", []

        context = self._format_context(results)
        prompt = QA_PROMPT_TEMPLATE.format(context_str=context, query_str=question)
        response = self._generate_response(prompt)
        sources = self._get_sources(results)

        return response, sources
