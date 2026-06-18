"""Multi-retriever system for the RAG pipeline."""

from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

from ..config import TOP_K_RESULTS


class MultiRetriever:
    """
    Multi-retriever system that combines results from different retrieval strategies.
    Currently supports:
    - Vector similarity search
    - Metadata filtering
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = TOP_K_RESULTS
    ):
        self.index = index
        self.top_k = top_k

        # Create vector retriever
        self.vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )

    def retrieve(
        self,
        query: str,
        source_type: Optional[str] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes for a query.

        Args:
            query: User's question
            source_type: Optional filter by source type (pdf, csv, web)
        """
        # Get results from vector search
        results = self.vector_retriever.retrieve(query)

        # Filter by source type if specified
        if source_type:
            results = [
                r for r in results
                if r.node.metadata.get("source_type") == source_type
            ]

        return results

    def retrieve_with_metadata_filter(
        self,
        query: str,
        metadata_filters: dict
    ) -> List[NodeWithScore]:
        """
        Retrieve with metadata filtering.

        Args:
            query: User's question
            metadata_filters: Dict of metadata key-value pairs to filter by
        """
        results = self.vector_retriever.retrieve(query)

        # Apply metadata filters
        filtered_results = []
        for result in results:
            match = True
            for key, value in metadata_filters.items():
                if result.node.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_results.append(result)

        return filtered_results

    def hybrid_retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Hybrid retrieval combining multiple strategies.
        Retrieves more results initially and relies on re-ranking.
        """
        top_k = top_k or self.top_k * 2

        # Create retriever with more results for re-ranking
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )

        results = retriever.retrieve(query)
        return results
