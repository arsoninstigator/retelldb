"""
Query classifier.
Detects whether an incoming query maps cleanly to known semantic tag nodes in the documentation graph.
Supports deterministic stemming and normalization across query tokens and tag names.
"""

from typing import List, Tuple, Set
import re
from retelldb.models.schema import GraphNode


def _stem_word(word: str) -> str:
    """
    Deterministic rule-based English suffix stemmer.
    Handles plurals, common verb/noun inflections without external dependencies.
    """
    w = word.lower().strip()
    if len(w) <= 3:
        return w

    # Plurals and standard suffixes
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        if w.endswith(("ses", "xes", "zes", "ches", "shes")):
            return w[:-2]
        return w[:-1]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 4:
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if w.endswith("ed") and len(w) > 3:
        stem = w[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if w.endswith("tion") and len(w) > 5:
        return w[:-4]
    if w.endswith("ment") and len(w) > 5:
        return w[:-4]
    return w


def _tokenize(text: str) -> List[str]:
    """Extract lowercased alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())


def _normalize_phrase(text: str) -> str:
    """Normalize whitespace and separators in phrases."""
    return re.sub(r"[\-_/\s]+", " ", text).strip().lower()


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
        q_norm = _normalize_phrase(query)
        q_tokens = _tokenize(query)
        q_stems = [_stem_word(t) for t in q_tokens]
        q_stem_set = set(q_stems)

        matched_tags: List[str] = []
        scores: List[float] = []

        for tag_node in graph_tags:
            tag_name = tag_node.title.lower().strip()
            if not tag_name:
                continue

            tag_norm = _normalize_phrase(tag_name)
            tag_tokens = _tokenize(tag_name)
            tag_stems = [_stem_word(t) for t in tag_tokens]

            # 1. Exact phrase/word match
            pattern = rf"\b{re.escape(tag_norm)}\b"
            if re.search(pattern, q_norm):
                matched_tags.append(tag_name)
                scores.append(1.0)
                continue

            # 2. Compact compound match (e.g. "setup" <-> "set up", "quickstart" <-> "quick start")
            tag_compact = tag_norm.replace(" ", "")
            q_compact = q_norm.replace(" ", "")
            if len(tag_compact) >= 4 and tag_compact in q_compact:
                matched_tags.append(tag_name)
                scores.append(0.95)
                continue

            # 3. Deterministic stemmed token match
            if tag_stems:
                if len(tag_stems) == 1:
                    stem = tag_stems[0]
                    if stem in q_stem_set:
                        matched_tags.append(tag_name)
                        scores.append(0.95)
                        continue
                    # Prefix / alias matching (e.g. tag "auth" matches "authentication")
                    if stem == "auth" and any(qs.startswith("auth") for qs in q_stems):
                        matched_tags.append(tag_name)
                        scores.append(0.95)
                        continue
                else:
                    # Multi-token tag: all stems must be present in query
                    if all(ts in q_stem_set for ts in tag_stems):
                        matched_tags.append(tag_name)
                        scores.append(0.95)
                        continue

            # 4. Fallback substring match for compound tags
            if len(tag_name) > 3 and tag_name in q_lower:
                matched_tags.append(tag_name)
                scores.append(0.8)

        if not matched_tags:
            return False, [], 0.0

        avg_confidence = sum(scores) / len(scores)
        has_signal = len(matched_tags) >= 1 and avg_confidence >= threshold

        return has_signal, matched_tags, avg_confidence
