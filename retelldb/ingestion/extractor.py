"""
Dual-tier LLM entity and relationship extractor.
Uses LiteLLM / Featherless with a fast model tier for tags/types and a stronger model tier
for ambiguous relationship inference (depends_on, implements, replaces, related).
"""

import json
import logging
import re
from typing import List, Dict, Tuple, Any, Optional
from retelldb.models.schema import NodeType, EdgeType, GraphNode, GraphEdge, generate_edge_id
from retelldb.ingestion.normalizers.base import CandidateNode, CandidateEdge
from retelldb.config import InferenceConfig, default_config

logger = logging.getLogger("retelldb.extractor")


class LLMExtractor:
    """Extracts refined tags, entity types, and semantic edges using the dual-tier model split."""

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or default_config.inference
        self._litellm_available = False
        try:
            import litellm
            self.litellm = litellm
            self._litellm_available = True
        except ImportError:
            self.litellm = None

    def refine_nodes(self, candidates: List[CandidateNode]) -> List[GraphNode]:
        """Runs the fast model tier to refine tags and entity typing for each candidate."""
        refined_nodes: List[GraphNode] = []

        for candidate in candidates:
            # Use fast model if available and API key present
            if self._litellm_available and self.config.api_key:
                try:
                    refined = self._refine_node_with_llm(candidate)
                    refined_nodes.append(refined)
                    continue
                except Exception as e:
                    logger.warning(f"Fast model LLM refinement failed: {e}. Falling back to heuristics.")

            # Fallback heuristic refinement
            refined_nodes.append(self._heuristic_refine_node(candidate))

        return refined_nodes

    def infer_relationships(
        self, 
        nodes: List[GraphNode], 
        candidate_edges: List[CandidateEdge]
    ) -> List[GraphEdge]:
        """Runs the stronger model tier to infer ambiguous relationships between nodes."""
        final_edges: List[GraphEdge] = []

        # Convert initial candidate edges into GraphEdges
        for ce in candidate_edges:
            final_edges.append(
                GraphEdge(
                    id=ce.id,
                    source_id=ce.source_id,
                    target_id=ce.target_id,
                    type=ce.type,
                    weight=ce.weight,
                    metadata=ce.metadata
                )
            )

        # Look for ambiguous relationships (e.g. cross-referencing nodes)
        if len(nodes) > 1:
            if self._litellm_available and self.config.api_key:
                try:
                    inferred = self._infer_edges_with_llm(nodes)
                    final_edges.extend(inferred)
                    return self._deduplicate_edges(final_edges)
                except Exception as e:
                    logger.warning(f"Strong model edge inference failed: {e}. Falling back to heuristics.")

            # Fallback heuristic edge inference
            heuristic_edges = self._heuristic_infer_edges(nodes)
            final_edges.extend(heuristic_edges)

        return self._deduplicate_edges(final_edges)

    def _refine_node_with_llm(self, candidate: CandidateNode) -> GraphNode:
        """Call fast model tier via LiteLLM for tag extraction and type verification."""
        max_chars = self.config.node_content_llm_max_chars
        prompt = (
            f"You are an ingestion classifier for documentation graphs.\n"
            f"Node Title: {candidate.title}\n"
            f"Node Content: {candidate.content[:max_chars]}\n"
            f"Allowed Node Types: concept, api_endpoint, sdk_method, example, guide, error\n"
            f"Return JSON with keys 'type' (one of the allowed types) and 'tags' (list of 2-5 concise lowercase semantic tags)."
        )
        response = self.litellm.completion(
            model=self.config.fast_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.config.api_key,
            temperature=0.1,
            max_tokens=150,
        )
        raw_text = response.choices[0].message.content
        data = self._extract_json(raw_text)
        
        node_type = candidate.type
        if data.get("type"):
            try:
                node_type = NodeType(data["type"].lower())
            except ValueError:
                pass
        
        tags = set(candidate.tags)
        if isinstance(data.get("tags"), list):
            tags.update(str(t).strip().lower() for t in data["tags"])

        return GraphNode(
            id=candidate.id,
            title=candidate.title,
            type=node_type,
            content=candidate.content,
            audience=candidate.audience,
            tags=sorted(list(tags)),
            source_uri=candidate.source_uri,
            locator=candidate.locator,
            metadata=candidate.metadata,
            sources=list(candidate.sources)
        )

    def _infer_edges_with_llm(self, nodes: List[GraphNode]) -> List[GraphEdge]:
        """Call stronger model tier via LiteLLM to detect depends_on, replaces, implements edges."""
        summaries = [f"ID: {n.id} | Title: {n.title} | Type: {n.type.value}" for n in nodes[:15]]
        prompt = (
            f"Analyze these documentation nodes and identify semantic relationships between them.\n"
            f"Nodes:\n" + "\n".join(summaries) + "\n\n"
            f"Allowed edge types: depends_on, implements, replaces, related\n"
            f"Return JSON array of objects: [{{'source_id': '...', 'target_id': '...', 'type': '...', 'weight': 0.9}}]"
        )
        response = self.litellm.completion(
            model=self.config.strong_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.config.api_key,
            temperature=0.2,
            max_tokens=400,
        )
        raw_text = response.choices[0].message.content
        data = self._extract_json(raw_text)
        
        edges: List[GraphEdge] = []
        if isinstance(data, list):
            node_ids = {n.id for n in nodes}
            for item in data:
                src = item.get("source_id")
                tgt = item.get("target_id")
                e_type = item.get("type")
                if src in node_ids and tgt in node_ids and src != tgt:
                    try:
                        e_enum = EdgeType(e_type.lower())
                        edges.append(
                            GraphEdge(
                                id=generate_edge_id(src, tgt, e_enum.value),
                                source_id=src,
                                target_id=tgt,
                                type=e_enum,
                                weight=float(item.get("weight", 0.8)),
                                metadata={"inferred_by": "strong_llm"}
                            )
                        )
                    except ValueError:
                        continue
        return edges

    def _heuristic_refine_node(self, candidate: CandidateNode) -> GraphNode:
        """Rule-based tag normalization and validation."""
        tags = set(candidate.tags)
        # Extract keywords from title
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", candidate.title.lower())
        stop_words = {"the", "and", "for", "with", "this", "that", "how", "what", "from", "using", "into", "page"}
        for w in words:
            if w not in stop_words:
                tags.add(w)

        return GraphNode(
            id=candidate.id,
            title=candidate.title,
            type=candidate.type,
            content=candidate.content,
            audience=candidate.audience,
            tags=sorted(list(tags)),
            source_uri=candidate.source_uri,
            locator=candidate.locator,
            metadata=candidate.metadata,
            sources=list(candidate.sources)
        )

    def _heuristic_infer_edges(self, nodes: List[GraphNode]) -> List[GraphEdge]:
        """Detect shared tags, version replacements (v1 -> v2), and SDK implementation links with deterministic IDs."""
        edges: List[GraphEdge] = []
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                # Check for version replacement
                if self._is_replacement(n1.title, n2.title):
                    edges.append(
                        GraphEdge(
                            id=generate_edge_id(n2.id, n1.id, EdgeType.REPLACES.value),
                            source_id=n2.id,
                            target_id=n1.id,
                            type=EdgeType.REPLACES,
                            weight=1.0,
                            metadata={"inferred_by": "version_heuristic"}
                        )
                    )
                # Check if SDK method implements API endpoint
                elif n1.type == NodeType.SDK_METHOD and n2.type == NodeType.API_ENDPOINT:
                    if set(n1.tags).intersection(set(n2.tags)):
                        edges.append(
                            GraphEdge(
                                id=generate_edge_id(n1.id, n2.id, EdgeType.IMPLEMENTS.value),
                                source_id=n1.id,
                                target_id=n2.id,
                                type=EdgeType.IMPLEMENTS,
                                weight=0.9,
                                metadata={"inferred_by": "sdk_implement_heuristic"}
                            )
                        )
                # High tag overlap -> related edge
                else:
                    shared_tags = set(n1.tags).intersection(set(n2.tags))
                    if len(shared_tags) >= 2 and n1.type != NodeType.TAG and n2.type != NodeType.TAG:
                        e_type = EdgeType.RELATED
                        edges.append(
                            GraphEdge(
                                id=generate_edge_id(n1.id, n2.id, e_type.value),
                                source_id=n1.id,
                                target_id=n2.id,
                                type=e_type,
                                weight=min(0.5 + 0.1 * len(shared_tags), 0.9),
                                metadata={"shared_tags": list(shared_tags)}
                            )
                        )
        return edges

    def _is_replacement(self, title1: str, title2: str) -> bool:
        v1 = re.search(r"\bv(\d+)\b", title1.lower())
        v2 = re.search(r"\bv(\d+)\b", title2.lower())
        if v1 and v2:
            return int(v2.group(1)) > int(v1.group(1))
        return False

    def _extract_json(self, text: str) -> Any:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1)
        try:
            return json.loads(text.strip())
        except Exception:
            return {}

    def _deduplicate_edges(self, edges: List[GraphEdge]) -> List[GraphEdge]:
        seen = set()
        unique = []
        for e in edges:
            key = (e.source_id, e.target_id, e.type.value)
            if key not in seen and e.source_id != e.target_id:
                seen.add(key)
                unique.append(e)
        return unique
