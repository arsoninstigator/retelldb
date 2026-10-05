"""
HTML documentation normalizer.
Extracts title, nav links, headings as structure nodes,
and pre/code blocks as example nodes with 'example_of' edges.
Generates deterministic IDs for idempotent re-ingestion.
"""

from html.parser import HTMLParser
import re
from typing import Optional, List, Dict, Tuple
from retelldb.ingestion.normalizers.base import BaseNormalizer, NormalizedDocument, CandidateNode, CandidateEdge
from retelldb.models.schema import NodeType, EdgeType, AudienceType


class _HTMLDocParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title: Optional[str] = None
        self.nav_links: List[Tuple[str, str]] = []  # (text, href)
        self.sections: List[Tuple[int, str, str]] = []  # (level, heading, text)
        self.code_blocks: List[Tuple[str, str]] = []  # (lang, code)
        
        self._current_tag: Optional[str] = None
        self._in_title = False
        self._in_nav = False
        self._current_heading_level: Optional[int] = None
        self._current_heading_text = ""
        self._current_section_text: List[str] = []
        self._in_code = False
        self._current_code_lang = "text"
        self._current_code: List[str] = []
        self._current_a_href: Optional[str] = None
        self._current_a_text = ""

    def handle_starttag(self, tag, attrs):
        self._current_tag = tag.lower()
        attr_dict = dict(attrs)
        
        if tag == "title":
            self._in_title = True
        elif tag in ("nav", "aside"):
            self._in_nav = True
        elif tag == "a" and self._in_nav:
            self._current_a_href = attr_dict.get("href")
            self._current_a_text = ""
        elif re.match(r"^h[1-6]$", tag):
            self._flush_section()
            self._current_heading_level = int(tag[1])
            self._current_heading_text = ""
        elif tag in ("pre", "code"):
            self._in_code = True
            classes = attr_dict.get("class", "")
            lang_match = re.search(r"language-([a-zA-Z0-9_\-]+)", classes)
            if lang_match:
                self._current_code_lang = lang_match.group(1)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag in ("nav", "aside"):
            self._in_nav = False
        elif tag == "a" and self._current_a_href:
            self.nav_links.append((self._current_a_text.strip(), self._current_a_href))
            self._current_a_href = None
            self._current_a_text = ""
        elif re.match(r"^h[1-6]$", tag):
            pass  # heading text collected
        elif tag in ("pre", "code"):
            if self._in_code and self._current_code:
                code_text = "".join(self._current_code)
                if len(code_text.strip()) > 5:
                    self.code_blocks.append((self._current_code_lang, code_text))
            self._in_code = False
            self._current_code = []
            self._current_code_lang = "text"

    def handle_data(self, data):
        if self._in_title:
            self.title = (self.title or "") + data
        elif self._in_code:
            self._current_code.append(data)
        elif self._current_a_href:
            self._current_a_text += data
        elif self._current_heading_level is not None:
            self._current_heading_text += data
        else:
            self._current_section_text.append(data)

    def _flush_section(self):
        if self._current_heading_text.strip():
            text = " ".join("".join(self._current_section_text).split())
            self.sections.append((
                self._current_heading_level or 1,
                self._current_heading_text.strip(),
                text
            ))
        self._current_heading_text = ""
        self._current_section_text = []

    def close(self):
        super().close()
        self._flush_section()


class HTMLNormalizer(BaseNormalizer):
    """Normalizes HTML documentation pages into structured graph candidate nodes and edges."""

    def normalize(
        self,
        content: str,
        audience: AudienceType = AudienceType.DEVELOPER,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        **kwargs
    ) -> NormalizedDocument:
        parser = _HTMLDocParser()
        parser.feed(content)
        parser.close()

        doc_title = title or parser.title or "HTML Documentation"

        nodes: List[CandidateNode] = []
        edges: List[CandidateEdge] = []
        global_tags: List[str] = ["html", "doc"]

        root_node = CandidateNode.create_deterministic(
            title=doc_title,
            type=NodeType.GUIDE,
            content=f"HTML Documentation Page: {doc_title}",
            audience=audience,
            tags=global_tags.copy(),
            source_uri=source_uri,
            doc_title=doc_title,
            locator="root",
            metadata={"source_uri": source_uri, "is_root": True}
        )
        nodes.append(root_node)

        # Build sections
        section_nodes: Dict[int, CandidateNode] = {0: root_node}

        for level, heading_text, section_body in parser.sections:
            clean_heading = heading_text.strip()
            if not clean_heading:
                continue

            node_type = self._detect_node_type(clean_heading, section_body)
            section_tags = list(set(global_tags + [clean_heading.lower().replace(" ", "-")]))
            locator = f"h{level}:{clean_heading}"

            section_node = CandidateNode.create_deterministic(
                title=clean_heading,
                type=node_type,
                content=section_body[:1500] if section_body else clean_heading,
                audience=audience,
                tags=section_tags,
                source_uri=source_uri,
                doc_title=doc_title,
                locator=locator,
                metadata={"level": level, "source_uri": source_uri}
            )
            nodes.append(section_node)

            parent_node = root_node
            for lvl in range(level - 1, 0, -1):
                if lvl in section_nodes:
                    parent_node = section_nodes[lvl]
                    break
            section_nodes[level] = section_node

            edges.append(
                CandidateEdge.create_deterministic(
                    source_id=section_node.id,
                    target_id=parent_node.id,
                    type=EdgeType.DEPENDS_ON,
                    weight=0.8
                )
            )

        # Process code blocks
        for idx, (lang, code) in enumerate(parser.code_blocks, 1):
            ex_locator = f"code-block:{idx}:{lang}"
            example_node = CandidateNode.create_deterministic(
                title=f"Code Sample {idx} ({lang})",
                type=NodeType.EXAMPLE,
                content=code.strip(),
                audience=audience,
                tags=list(set(global_tags + [lang.lower()])),
                source_uri=source_uri,
                doc_title=doc_title,
                locator=ex_locator,
                metadata={"language": lang}
            )
            nodes.append(example_node)
            edges.append(
                CandidateEdge.create_deterministic(
                    source_id=example_node.id,
                    target_id=root_node.id,
                    type=EdgeType.EXAMPLE_OF,
                    weight=1.0
                )
            )

        return NormalizedDocument(
            document_title=doc_title,
            source_uri=source_uri,
            audience=audience,
            nodes=nodes,
            edges=edges,
            global_tags=global_tags
        )

    def _detect_node_type(self, heading: str, body: str) -> NodeType:
        h_lower = heading.lower()
        if any(w in h_lower for w in ["api", "endpoint", "route", "post", "get", "put"]):
            return NodeType.API_ENDPOINT
        if any(w in h_lower for w in ["method", "sdk", "function"]):
            return NodeType.SDK_METHOD
        if any(w in h_lower for w in ["error", "status", "troubleshoot"]):
            return NodeType.ERROR
        if any(w in h_lower for w in ["guide", "tutorial", "walkthrough", "overview"]):
            return NodeType.GUIDE
        return NodeType.CONCEPT
