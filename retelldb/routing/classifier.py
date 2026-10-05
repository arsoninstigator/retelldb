"""
Query classifier.
Detects whether an incoming query maps cleanly to known semantic tag nodes in the documentation graph.
"""

from typing import List, Tuple
import re
from retelldb.models.schema import GraphNode


class QueryClassifier:
    """Evaluates query token matches against graph tag taxonomy."""

    def evaluate_tag_signal(
        self,
        query: str,
        graph_tags: List[GraphNode],
        threshold: float = 0.6
    ) -> Tuple[bool, List[str], float]:
        """
        Check if the query matches known tag nodes with high confidence.
        Returns: (has_strong_tag_signal, matched_tag_names, confidence_score)
        """
        if not graph_tags:
            return False, [], 0.0

        q_lower = query.lower()
        matched_tags: List[str] = []
        scores: List[float] = []

        for tag_node in graph_tags:
            tag_name = tag_node.title.lower().strip()
            if not tag_name:
                continue

            # Exact word match
            pattern = rf"\b{re.escape(tag_name)}\b"
            if re.search(pattern, q_lower):
                matched_tags.append(tag_name)
                scores.append(1.0)
            elif len(tag_name) > 3 and tag_name in q_lower:
                # Substring match for compound tags like "token-refresh" or "auth"
                matched_tags.append(tag_name)
                scores.append(0.8)

        if not matched_tags:
            return False, [], 0.0

        avg_confidence = sum(scores) / len(scores)
        has_signal = len(matched_tags) >= 1 and avg_confidence >= threshold

        return has_signal, matched_tags, avg_confidence
