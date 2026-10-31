"""
Source citations.

Retrieval should hand the agent the source material, not a summary of it, and every
citation should be checkable. A node can carry several source references (a doc
section, a code block, an OpenAPI operation); the engine resolves them into
citations and, when the source text is available, verifies that the cited line
range actually contains the content being cited.
"""

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from retelldb.models.schema import GraphNode, SourceRef

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


class Citation(BaseModel):
    """One resolved reference to where a node came from."""

    node_id: str
    node_title: str
    uri: Optional[str] = None
    path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    content_type: str = "text"
    excerpt: str = ""
    # True when the cited span was checked against the source, None when unverifiable
    verified: Optional[bool] = None

    def location(self) -> str:
        """Compact human-readable location, e.g. docs/auth.md:45-62."""
        target = self.path or self.uri or "unknown"
        if self.line_start is None:
            return target
        if self.line_end is None or self.line_end == self.line_start:
            return f"{target}:{self.line_start}"
        return f"{target}:{self.line_start}-{self.line_end}"


class SourceVerifier:
    """
    Resolves and verifies citations against source text.

    Pass a mapping of uri/path -> raw source text. Anything not in the mapping is
    reported as unverified rather than assumed correct.
    """

    def __init__(self, sources: Optional[Dict[str, str]] = None):
        self.sources = sources or {}

    def text_for(self, citation: Citation) -> Optional[str]:
        for key in (citation.uri, citation.path):
            if key and key in self.sources:
                return self.sources[key]
        return None

    def cited_span(self, citation: Citation) -> Optional[str]:
        text = self.text_for(citation)
        if text is None or citation.line_start is None:
            return None
        lines = text.splitlines()
        start = max(0, citation.line_start - 1)
        end = citation.line_end if citation.line_end else citation.line_start
        return "\n".join(lines[start:end])

    def verify(self, citation: Citation) -> Optional[bool]:
        """
        True when the cited span contains the content being cited.

        The check is deliberately about overlap, not equality: a heading node cites
        its heading line, and a section node cites the section it was cut from.
        Returns None when the source text is unavailable.
        """
        span = self.cited_span(citation)
        if span is None:
            return None
        span_norm = _normalize(span)
        excerpt_norm = _normalize(citation.excerpt)
        if not span_norm or not excerpt_norm:
            return False
        needle = excerpt_norm[:160]
        if needle in span_norm:
            return True
        # very short node content ("401", "auth"): check the other direction
        if len(needle) < 20 and needle and needle in span_norm:
            return True
        return False


def citations_for_node(
    node: GraphNode,
    verifier: Optional[SourceVerifier] = None,
    max_excerpt: int = 320,
) -> List[Citation]:
    """Resolve every source reference on a node into a citation."""
    refs: List[SourceRef] = list(node.sources or [])

    # Nodes ingested before sources existed still deserve a citation
    if not refs and (node.source_uri or node.locator):
        refs = [
            SourceRef(
                uri=node.source_uri,
                path=node.source_uri,
                content_type="markdown",
                excerpt=node.content[:max_excerpt],
            )
        ]

    citations: List[Citation] = []
    for ref in refs:
        excerpt = (ref.excerpt or node.content or "")[:max_excerpt]
        citation = Citation(
            node_id=node.id,
            node_title=node.title,
            uri=ref.uri or node.source_uri,
            path=ref.path or ref.uri or node.source_uri,
            line_start=ref.line_start,
            line_end=ref.line_end,
            content_type=ref.content_type or "text",
            excerpt=excerpt,
        )
        if verifier is not None:
            citation.verified = verifier.verify(citation)
        citations.append(citation)
    return citations
