"""Re-ranking module for improving retrieval relevance."""

from typing import List, Optional
from llama_index.core.schema import NodeWithScore

# Optional: Use sentence-transformers for cross-encoder reranking
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


class SemanticReranker:
    """Re-rank results using a cross-encoder model for better relevance."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if not RERANKER_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: List[NodeWithScore],
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Re-rank results based on semantic relevance to query.

        Args:
            query: User's question
            results: Initial retrieval results
            top_k: Number of results to return after reranking
        """
        if not results:
            return results

        top_k = top_k or len(results)

        # Prepare query-document pairs
        pairs = [(query, r.node.get_content()) for r in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Update scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)

        # Sort by new scores (higher is better for cross-encoder)
        reranked = sorted(results, key=lambda x: x.score, reverse=True)

        return reranked[:top_k]


def rerank_results(
    query: str,
    results: List[NodeWithScore],
    top_k: Optional[int] = None,
    use_semantic_reranking: bool = True
) -> List[NodeWithScore]:
    """
    Re-rank retrieval results.

    Args:
        query: User's question
        results: Initial retrieval results
        top_k: Number of results to return
        use_semantic_reranking: Whether to use cross-encoder reranking
    """
    if not results:
        return results

    top_k = top_k or 5

    if use_semantic_reranking and RERANKER_AVAILABLE:
        try:
            reranker = SemanticReranker()
            return reranker.rerank(query, results, top_k)
        except Exception as e:
            print(f"Semantic reranking failed: {e}. Using score-based ranking.")

    # Fallback: sort by original similarity score
    sorted_results = sorted(results, key=lambda x: x.score or 0, reverse=True)
    return sorted_results[:top_k]


def deduplicate_results(results: List[NodeWithScore], threshold: float = 0.9) -> List[NodeWithScore]:
    """Remove near-duplicate results based on text similarity."""
    if len(results) <= 1:
        return results

    unique_results = [results[0]]

    for result in results[1:]:
        is_duplicate = False
        result_text = result.node.get_content().lower()

        for unique in unique_results:
            unique_text = unique.node.get_content().lower()
            # Simple overlap check
            overlap = len(set(result_text.split()) & set(unique_text.split()))
            max_len = max(len(result_text.split()), len(unique_text.split()))

            if max_len > 0 and overlap / max_len > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_results.append(result)

    return unique_results
