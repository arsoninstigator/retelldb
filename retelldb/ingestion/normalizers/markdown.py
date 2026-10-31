"""
Markdown and MDX normalizer.
Extracts frontmatter tags, hierarchical headings into structure nodes,
and code blocks into typed example nodes with 'example_of' edges.
Generates deterministic IDs for idempotent re-ingestion.
"""

import re
import yaml
from typing import Optional, List, Dict, Tuple
from retelldb.ingestion.normalizers.base import BaseNormalizer, NormalizedDocument, CandidateNode, CandidateEdge
from retelldb.models.schema import NodeType, EdgeType, AudienceType, SourceRef


class MarkdownNormalizer(BaseNormalizer):
    """Parses Markdown and MDX documents into structured candidate nodes and edges."""

    def normalize(
        self,
        content: str,
        audience: AudienceType = AudienceType.DEVELOPER,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        **kwargs
    ) -> NormalizedDocument:
        frontmatter, md_body, body_line_offset = self._extract_frontmatter(content)
        
        # Tags from frontmatter
        fm_tags = []
        if isinstance(frontmatter.get("tags"), list):
            fm_tags = [str(t).strip().lower() for t in frontmatter["tags"]]
        elif isinstance(frontmatter.get("tags"), str):
            fm_tags = [t.strip().lower() for t in frontmatter["tags"].split(",") if t.strip()]

        doc_title = title or frontmatter.get("title") or self._extract_first_h1(md_body) or "Untitled Document"
        
        # Override audience if present in frontmatter
        resolved_audience = audience
        if "audience" in frontmatter:
            aud_str = str(frontmatter["audience"]).lower()
            if aud_str in ("user", "end_user"):
                resolved_audience = AudienceType.USER
            elif aud_str in ("dev", "developer"):
                resolved_audience = AudienceType.DEVELOPER

        nodes: List[CandidateNode] = []
        edges: List[CandidateEdge] = []

        # Root document node (guide or concept)
        root_node = CandidateNode.create_deterministic(
            title=doc_title,
            type=NodeType.GUIDE if "guide" in doc_title.lower() or "tutorial" in doc_title.lower() else NodeType.CONCEPT,
            content=f"Document: {doc_title}",
            audience=resolved_audience,
            tags=fm_tags.copy(),
            source_uri=source_uri,
            doc_title=doc_title,
            locator="root",
            metadata={"source_uri": source_uri, "is_root": True, **frontmatter},
            sources=[
                SourceRef(
                    uri=source_uri,
                    path=source_uri,
                    line_start=1,
                    line_end=max(1, body_line_offset + len(md_body.splitlines())),
                    content_type="markdown",
                    excerpt=md_body.strip()[:200],
                )
            ],
        )
        nodes.append(root_node)

        # Parse sections by headings
        sections = self._split_by_headings(md_body, line_offset=body_line_offset)
        
        section_nodes: Dict[int, CandidateNode] = {0: root_node}  # level -> last node at this level

        for level, heading_text, section_body, head_line, end_line in sections:
            clean_heading = heading_text.strip()
            if not clean_heading:
                continue

            node_type = self._detect_node_type(clean_heading, section_body)
            # body starts on the line after the heading
            code_blocks, text_without_code = self._extract_code_blocks(
                section_body, body_start_line=head_line + 1
            )
            section_tags = list(set(fm_tags + self._extract_inline_tags(clean_heading)))
            locator = f"heading-lvl{level}:{clean_heading}"
            
            section_node = CandidateNode.create_deterministic(
                title=clean_heading,
                type=node_type,
                content=text_without_code.strip() or clean_heading,
                audience=resolved_audience,
                tags=section_tags,
                source_uri=source_uri,
                doc_title=doc_title,
                locator=locator,
                metadata={"level": level, "source_uri": source_uri},
                sources=[
                    SourceRef(
                        uri=source_uri,
                        path=source_uri,
                        line_start=head_line,
                        line_end=end_line,
                        content_type="markdown",
                        excerpt=(text_without_code.strip() or clean_heading)[:200],
                    )
                ],
            )
            nodes.append(section_node)

            # Find parent in hierarchy
            parent_node = root_node
            for lvl in range(level - 1, 0, -1):
                if lvl in section_nodes:
                    parent_node = section_nodes[lvl]
                    break
            
            section_nodes[level] = section_node

            # Connect parent to section
            edges.append(
                CandidateEdge.create_deterministic(
                    source_id=section_node.id,
                    target_id=parent_node.id,
                    type=EdgeType.DEPENDS_ON if node_type != NodeType.ERROR else EdgeType.RELATED,
                    weight=0.8
                )
            )

            # Process code examples
            for idx, (lang, code, code_start, code_end) in enumerate(code_blocks, 1):
                example_title = f"{clean_heading} - Example {idx} ({lang})"
                example_locator = f"{locator}:example-{idx}:{lang}"
                example_node = CandidateNode.create_deterministic(
                    title=example_title,
                    type=NodeType.EXAMPLE,
                    content=code.strip(),
                    audience=resolved_audience,
                    tags=list(set(section_tags + [lang.lower()] if lang else section_tags)),
                    source_uri=source_uri,
                    doc_title=doc_title,
                    locator=example_locator,
                    metadata={"language": lang, "parent_heading": clean_heading},
                    sources=[
                        SourceRef(
                            uri=source_uri,
                            path=source_uri,
                            line_start=code_start,
                            line_end=code_end,
                            content_type="code",
                            excerpt=code.strip()[:200],
                        )
                    ],
                )
                nodes.append(example_node)

                # Connect example to parent section with example_of edge
                edges.append(
                    CandidateEdge.create_deterministic(
                        source_id=example_node.id,
                        target_id=section_node.id,
                        type=EdgeType.EXAMPLE_OF,
                        weight=1.0
                    )
                )

        return NormalizedDocument(
            document_title=doc_title,
            source_uri=source_uri,
            audience=resolved_audience,
            nodes=nodes,
            edges=edges,
            global_tags=fm_tags
        )

    def _extract_frontmatter(self, text: str) -> Tuple[Dict, str, int]:
        """
        Return (frontmatter, body, body_line_offset).

        The offset keeps citation line numbers pointing at the real file rather than
        at the body after frontmatter was stripped.
        """
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if match:
            fm_text = match.group(1)
            body = text[match.end():]
            offset = text.count("\n", 0, match.end())
            try:
                data = yaml.safe_load(fm_text) or {}
                return data, body, offset
            except Exception:
                return {}, body, offset
        return {}, text, 0

    def _extract_first_h1(self, text: str) -> Optional[str]:
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _split_by_headings(
        self, text: str, line_offset: int = 0
    ) -> List[Tuple[int, str, str, int, int]]:
        """
        Split a document into (level, heading, body, heading_line, end_line).

        Line numbers are 1-based and refer to the body passed in, so a citation can
        point at the exact section it came from.
        """
        heading_regex = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_regex.finditer(text))
        total_lines = max(1, len(text.splitlines()))

        if not matches:
            return [(1, "Content", text, 1 + line_offset, total_lines + line_offset)]

        sections = []
        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end]
            head_line = text.count("\n", 0, match.start()) + 1 + line_offset
            if i + 1 < len(matches):
                end_line = max(
                    head_line,
                    text.count("\n", 0, matches[i + 1].start()) + line_offset,
                )
            else:
                end_line = total_lines + line_offset
            sections.append((level, heading, body, head_line, end_line))

        return sections

    def _extract_code_blocks(
        self, text: str, body_start_line: int = 1
    ) -> Tuple[List[Tuple[str, str, int, int]], str]:
        code_regex = re.compile(r"```([a-zA-Z0-9_\-\.]*)\n(.*?)```", re.DOTALL)
        blocks = []
        for match in code_regex.finditer(text):
            lang = match.group(1).strip() or "text"
            code = match.group(2)
            start_line = body_start_line + text.count("\n", 0, match.start())
            end_line = start_line + max(1, code.count("\n"))
            blocks.append((lang, code, start_line, end_line))

        cleaned_text = code_regex.sub("[Code Example]", text)
        return blocks, cleaned_text

    def _detect_node_type(self, heading: str, body: str) -> NodeType:
        h_lower = heading.lower()
        if re.search(r"\b(get|post|put|delete|patch|options)\s+/[a-zA-Z0-9_\-/]*", heading, re.IGNORECASE):
            return NodeType.API_ENDPOINT
        if any(w in h_lower for w in ["endpoint", "rest api", "route", "http method"]):
            return NodeType.API_ENDPOINT
        if any(w in h_lower for w in ["method", "function", "class", "sdk", "client.", "def "]):
            return NodeType.SDK_METHOD
        if any(w in h_lower for w in ["error", "exception", "failure", "troubleshooting", "status code", "400", "401", "404", "500"]):
            return NodeType.ERROR
        if any(w in h_lower for w in ["guide", "tutorial", "walkthrough", "quickstart", "getting started"]):
            return NodeType.GUIDE
        return NodeType.CONCEPT

    def _extract_inline_tags(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", text.lower())
        stop_words = {"the", "and", "for", "with", "this", "that", "how", "what", "from", "using", "into"}
        return [w for w in words if w not in stop_words][:4]
