"""
Query classifier.
Detects whether an incoming query maps cleanly to known semantic tag nodes in the documentation graph.
Supports deterministic stemming and normalization across query tokens and tag names.
"""

from typing import List, Optional, Tuple, Set
import re
from pydantic import BaseModel, Field

from retelldb.models.schema import GraphNode


class TagMatch(BaseModel):
    """One auditable reason a query matched a tag node."""

    tag_name: str
    score: float
    # exact | compact | stem | prefix | substring: how the match happened
    rule: str = "exact"
    node_id: Optional[str] = None


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


def _is_prefix_match(stem: str, query_stems: List[str], min_shared: int = 4) -> bool:
    """True when a tag stem is a real prefix of a query stem (auth -> authentication)."""
    if len(stem) < min_shared:
        return False
    return any(qs != stem and qs.startswith(stem) for qs in query_stems)


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
        matches = self.match_tags(query, graph_tags)
        if not matches:
            return False, [], 0.0
        avg_confidence = sum(m.score for m in matches) / len(matches)
        has_signal = len(matches) >= 1 and avg_confidence >= threshold
        return has_signal, [m.tag_name for m in matches], avg_confidence

    def match_tags(
        self,
        query: str,
        graph_tags: List[GraphNode],
    ) -> List[TagMatch]:
        """
        Return every tag match with the rule that produced it.

        Each rule is explicit so a match can be explained. Nothing here silently
        rewrites the query or the tag.
        """
        if not graph_tags:
            return []

        q_lower = query.lower()
        q_norm = _normalize_phrase(query)
        q_tokens = _tokenize(query)
        q_stems = [_stem_word(t) for t in q_tokens]
        q_stem_set = set(q_stems)

        matches: List[TagMatch] = []
        seen: Set[str] = set()

        def add(tag_node: GraphNode, tag_name: str, score: float, rule: str) -> None:
            if tag_name in seen:
                return
            seen.add(tag_name)
            matches.append(
                TagMatch(
                    tag_name=tag_name,
                    score=score,
                    rule=rule,
                    node_id=tag_node.id,
                )
            )

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
                add(tag_node, tag_name, 1.0, "exact")
                continue

            # 2. Compact compound match (e.g. "setup" <-> "set up", "quickstart" <-> "quick start")
            tag_compact = tag_norm.replace(" ", "")
            q_compact = q_norm.replace(" ", "")
            if len(tag_compact) >= 4 and tag_compact in q_compact:
                add(tag_node, tag_name, 0.95, "compact")
                continue

            # 3. Deterministic stemmed token match
            if tag_stems:
                if len(tag_stems) == 1:
                    stem = tag_stems[0]
                    if stem in q_stem_set:
                        add(tag_node, tag_name, 0.95, "stem")
                        continue
                    # Prefix matching (e.g. tag "auth" matches "authentication"). This is
                    # a last-resort heuristic, labelled as such rather than passed off as
                    # a real alias: configured aliases live in the tag taxonomy.
                    if _is_prefix_match(stem, q_stems):
                        add(tag_node, tag_name, 0.85, "prefix")
                        continue
                else:
                    # Multi-token tag: all stems must be present in query
                    if all(ts in q_stem_set for ts in tag_stems):
                        add(tag_node, tag_name, 0.95, "stem")
                        continue

            # 4. Fallback substring match for compound tags
            if len(tag_name) > 3 and tag_name in q_lower:
                add(tag_node, tag_name, 0.8, "substring")

        return matches
