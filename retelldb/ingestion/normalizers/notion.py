"""
Notion export normalizer.
Parses Notion block JSON or exported Notion Markdown structures,
turning Notion pages into concept/guide nodes and links into candidate relationship edges
with deterministic IDs.
"""

import json
from typing import Optional, List, Dict, Any
from retelldb.ingestion.normalizers.base import BaseNormalizer, NormalizedDocument, CandidateNode, CandidateEdge
from retelldb.ingestion.normalizers.markdown import MarkdownNormalizer
from retelldb.models.schema import NodeType, EdgeType, AudienceType


class NotionNormalizer(BaseNormalizer):
    """Normalizes Notion block structures (or Notion exported markdown) into candidate nodes and edges."""

    def normalize(
        self,
        content: str,
        audience: AudienceType = AudienceType.DEVELOPER,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        **kwargs
    ) -> NormalizedDocument:
        # Check if content is Notion JSON block dump
        try:
            data = json.loads(content)
            if isinstance(data, dict) and ("results" in data or "object" in data or "type" in data):
                return self._normalize_notion_json(data, audience, title, source_uri)
        except Exception:
            pass

        # Fallback: Notion Markdown export
        md_norm = MarkdownNormalizer()
        doc = md_norm.normalize(content, audience, title, source_uri, **kwargs)
        for node in doc.nodes:
            node.tags.append("notion")
        doc.global_tags.append("notion")
        return doc

    def _normalize_notion_json(
        self, 
        data: Dict[str, Any], 
        audience: AudienceType, 
        title: Optional[str], 
        source_uri: Optional[str]
    ) -> NormalizedDocument:
        doc_title = title or "Notion Page"
        nodes: List[CandidateNode] = []
        edges: List[CandidateEdge] = []
        global_tags: List[str] = ["notion"]

        root_node = CandidateNode.create_deterministic(
            title=doc_title,
            type=NodeType.GUIDE,
            content=f"Notion Page: {doc_title}",
            audience=audience,
            tags=global_tags.copy(),
            source_uri=source_uri,
            doc_title=doc_title,
            locator="root",
            metadata={"source_uri": source_uri, "is_root": True}
        )
        nodes.append(root_node)

        blocks = data.get("results", [data]) if isinstance(data.get("results"), list) else [data]
        current_parent = root_node

        for idx, block in enumerate(blocks, 1):
            b_type = block.get("type", "paragraph")
            b_data = block.get(b_type, {})
            text_runs = b_data.get("rich_text", [])
            plain_text = "".join(r.get("plain_text", "") for r in text_runs).strip()
            
            if not plain_text:
                continue

            block_id = block.get("id") or f"block-{idx}"

            if b_type in ("heading_1", "heading_2", "heading_3"):
                h_node = CandidateNode.create_deterministic(
                    title=plain_text,
                    type=NodeType.CONCEPT,
                    content=plain_text,
                    audience=audience,
                    tags=list(set(global_tags + [plain_text.lower().replace(" ", "-")])),
                    source_uri=source_uri,
                    doc_title=doc_title,
                    locator=f"heading:{block_id}",
                    metadata={"notion_block_id": block_id}
                )
                nodes.append(h_node)
                edges.append(
                    CandidateEdge.create_deterministic(
                        source_id=h_node.id,
                        target_id=root_node.id,
                        type=EdgeType.DEPENDS_ON,
                        weight=0.8
                    )
                )
                current_parent = h_node
            elif b_type == "code":
                lang = b_data.get("language", "text")
                example_node = CandidateNode.create_deterministic(
                    title=f"Code Example ({lang})",
                    type=NodeType.EXAMPLE,
                    content=plain_text,
                    audience=audience,
                    tags=list(set(global_tags + [lang.lower()])),
                    source_uri=source_uri,
                    doc_title=doc_title,
                    locator=f"code:{block_id}:{lang}",
                    metadata={"language": lang}
                )
                nodes.append(example_node)
                edges.append(
                    CandidateEdge.create_deterministic(
                        source_id=example_node.id,
                        target_id=current_parent.id,
                        type=EdgeType.EXAMPLE_OF,
                        weight=1.0
                    )
                )
            else:
                current_parent.content += f"\n{plain_text}"

        return NormalizedDocument(
            document_title=doc_title,
            source_uri=source_uri,
            audience=audience,
            nodes=nodes,
            edges=edges,
            global_tags=global_tags
        )
