"""
RetellDB Routing Engine.

Orchestrates audience classification, tag resolution, weighted graph traversal,
associative vector fallback, perspective scoping, gap detection, confidence
tiering, and citation resolution.

The routing decision (tag traversal vs vector fallback vs hybrid) stays internal,
as the public contract promises. What the agent receives is the subgraph, the paths
that pulled each node in, where the content came from, and how confident the engine
actually is.
"""

import logging
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from retelldb.config import RetellDBConfig, default_config
from retelldb.models.query import (
    QueryRequest,
    QueryStrategy,
    ScoredNode,
    SubGraphResult,
)
from retelldb.models.schema import (
    AudienceType,
    GraphEdge,
    GraphNode,
    NodeType,
)
from retelldb.routing.audience import AudienceClassifier
from retelldb.routing.classifier import QueryClassifier, TagMatch
from retelldb.storage.base import BaseGraphStore, BaseVectorStore
from retelldb.storage.embedder import ContextualEmbedder
from retelldb.wrangling.citation import Citation, SourceVerifier, citations_for_node
from retelldb.wrangling.confidence import build_breakdown
from retelldb.wrangling.scope import (
    PersonaScope,
    ScopeBlock,
    evaluate_node,
)
from retelldb.wrangling.taxonomy import (
    ResolutionRule,
    TagResolution,
    TagTaxonomy,
    taxonomy_from_config,
)
from retelldb.wrangling.traversal import TraversedPath, WeightedTraverser

logger = logging.getLogger("retelldb.routing")


def source_confidence(node: GraphNode) -> float:
    """
    How well the node's origin is documented.

    A precise line range is stronger evidence than a bare blob URL, which is stronger
    than a node with no source at all. This feeds the confidence tier.
    """
    if not node.sources:
        return 0.6
    best = 0.6
    for ref in node.sources:
        if ref.line_start is not None:
            best = max(best, 1.0)
        elif ref.uri or ref.path:
            best = max(best, 0.85)
    return best


class RetrievalEngine:
    """The central query routing and retrieval engine of RetellDB."""

    def __init__(
        self,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        config: Optional[RetellDBConfig] = None,
        taxonomy: Optional[TagTaxonomy] = None,
        source_verifier: Optional[SourceVerifier] = None,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.config = config or default_config

        self.audience_classifier = AudienceClassifier(self.config.inference)
        self.query_classifier = QueryClassifier()
        self.embedder = ContextualEmbedder(self.config.inference)
        self.taxonomy = taxonomy or taxonomy_from_config(self.config)
        self.source_verifier = source_verifier

        wrangling = getattr(self.config, "wrangling", None)
        self.hopp_decay = getattr(wrangling, "hop_decay", 0.85)
        self.dampener_floor = getattr(wrangling, "dampener_floor", 0.45)
        self.min_edge_weight = getattr(wrangling, "min_edge_weight", 0.05)

    # -- main entry point -------------------------------------------------

    def retrieve(self, request: QueryRequest) -> SubGraphResult:
        """Execute the full routing and retrieval lifecycle for an agent query."""
        t_start = time.perf_counter()

        # 1. Audience routing
        audience = self.audience_classifier.classify(request.query, request.audience)
        target_graph = (
            self.config.storage.dev_graph_name
            if audience == AudienceType.DEVELOPER
            else self.config.storage.user_graph_name
        )

        # 2. Perspective
        scope = PersonaScope(
            persona_id=request.persona,
            allow_shared=request.allow_shared,
        )
        node_cache: Dict[str, Optional[GraphNode]] = {}
        blocked: Dict[str, ScopeBlock] = {}
        relevance: Dict[str, float] = {}

        def get_node(node_id: str) -> Optional[GraphNode]:
            if node_id not in node_cache:
                node_cache[node_id] = self.graph_store.get_node(target_graph, node_id)
            return node_cache[node_id]

        def in_scope(node_id: str) -> bool:
            node = get_node(node_id)
            if node is None:
                return False
            visible, reason, rel = evaluate_node(node, scope)
            if visible:
                relevance[node_id] = rel
                return True
            if node.type != NodeType.TAG and node_id not in blocked:
                blocked[node_id] = ScopeBlock(
                    node_id=node.id,
                    title=node.title,
                    reason=reason,
                    available_personas=sorted(
                        {p.strip().lower() for p in (node.personas or []) if p.strip()}
                    ),
                )
            return False

        # 3. Tag resolution: which tag nodes does this query actually hit?
        graph_tags = self.graph_store.get_tag_nodes(target_graph)
        matches: List[TagMatch] = self.query_classifier.match_tags(request.query, graph_tags)
        tag_confidence = (
            sum(m.score for m in matches) / len(matches) if matches else 0.0
        )
        has_signal = bool(matches) and tag_confidence >= request.tag_confidence_threshold
        resolutions = self._resolutions_for(matches, graph_tags)

        nodes: Dict[str, ScoredNode] = {}
        paths: Dict[str, TraversedPath] = {}
        traversed_edges: List[GraphEdge] = []
        dampeners: Dict[str, Dict[str, float]] = {}

        if has_signal:
            nodes, paths, traversed_edges, dampeners = self._tag_traversal(
                target_graph=target_graph,
                matches=matches,
                tag_confidence=tag_confidence,
                max_hops=request.max_hops,
                in_scope=in_scope,
                relevance=relevance,
                get_node=get_node,
                min_edge_weight=request.min_edge_weight,
            )
            strategy = QueryStrategy.TAG_TRAVERSAL

            # 4. Coverage gaps: tag traversal alone was not enough
            if self._detect_coverage_gaps(
                query=request.query,
                retrieved_nodes=list(nodes.values()),
                top_k=request.top_k,
            ):
                strategy = QueryStrategy.HYBRID
                remaining = max(2, request.top_k - len(nodes))
                for gap_node in self._execute_vector_search(
                    query=request.query,
                    target_graph=target_graph,
                    exclude_ids=set(nodes.keys()),
                    top_k=remaining,
                    in_scope=in_scope,
                ):
                    gap_node.retrieval_source = "gap_fill"
                    nodes[gap_node.node.id] = gap_node
                    dampeners[gap_node.node.id] = self._vector_dampeners(
                        gap_node.node, scope
                    )
        else:
            # 5. Latent associative fallback
            strategy = QueryStrategy.VECTOR_FALLBACK
            for vector_node in self._execute_vector_search(
                query=request.query,
                target_graph=target_graph,
                exclude_ids=set(),
                top_k=request.top_k,
                in_scope=in_scope,
            ):
                nodes[vector_node.node.id] = vector_node
                dampeners[vector_node.node.id] = self._vector_dampeners(
                    vector_node.node, scope
                )

        # 6. Rank, then collapse duplicate titles from separate source documents
        ranked = sorted(
            nodes.values(), key=lambda sn: (sn.score, sn.confidence), reverse=True
        )
        top_nodes = self._dedupe(ranked)[: request.top_k]

        # 7. Assemble edges, paths and citations for the returned subgraph
        final_edges: List[GraphEdge] = []
        if request.include_edges and top_nodes:
            top_ids = [sn.node.id for sn in top_nodes]
            subgraph_edges = self.graph_store.get_subgraph_between(target_graph, top_ids)
            top_set = set(top_ids)
            edge_map: Dict[str, GraphEdge] = {}
            for edge in subgraph_edges:
                edge_map[edge.id] = edge
            for edge in traversed_edges:
                if edge.source_id in top_set or edge.target_id in top_set:
                    edge_map.setdefault(edge.id, edge)
            final_edges = list(edge_map.values())

        top_paths = [
            paths[sn.node.id].describe()
            for sn in top_nodes
            if sn.node.id in paths
        ]

        citations: List[Citation] = []
        if request.include_citations:
            for scored in top_nodes:
                citations.extend(
                    citations_for_node(scored.node, verifier=self.source_verifier)
                )

        # 8. Confidence: total, tier, and what dampened it
        confidence = self._confidence_for(
            base_signal=tag_confidence,
            top_node=top_nodes[0] if top_nodes else None,
            dampeners=dampeners,
            blocked=blocked,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return SubGraphResult(
            query=request.query,
            target_graph=target_graph,
            retrieval_path=strategy,
            nodes=top_nodes,
            edges=final_edges,
            traversal_chains=self._build_traversal_chains(
                [sn.node.id for sn in top_nodes], final_edges
            ),
            confidence_signals={
                "tag_confidence": round(tag_confidence, 4),
                "matched_tags_count": float(len(matches)),
                "avg_node_score": (
                    round(sum(n.score for n in top_nodes) / len(top_nodes), 4)
                    if top_nodes
                    else 0.0
                ),
            },
            execution_time_ms=round(elapsed_ms, 2),
            confidence=confidence,
            paths=top_paths,
            tag_resolutions=resolutions,
            blocked=list(blocked.values()) if request.include_blocked else [],
            citations=citations,
        )

    # -- tag traversal ----------------------------------------------------

    def _tag_traversal(
        self,
        target_graph: str,
        matches: List[TagMatch],
        tag_confidence: float,
        max_hops: int,
        in_scope: Callable[[str], bool],
        relevance: Dict[str, float],
        get_node: Callable[[str], Optional[GraphNode]],
        min_edge_weight: Optional[float] = None,
    ) -> Tuple[
        Dict[str, ScoredNode],
        Dict[str, TraversedPath],
        List[GraphEdge],
        Dict[str, Dict[str, float]],
    ]:
        """
        Walk outward from every matched tag node, honouring edge weights.

        Each matched tag is walked separately so a node tagged with several matching
        tags is rewarded for the overlap, not just for being close to one of them.
        """
        traverser = WeightedTraverser(
            min_edge_weight=(
                self.min_edge_weight if min_edge_weight is None else min_edge_weight
            )
        )

        best_path: Dict[str, TraversedPath] = {}
        direct_tags: Dict[str, Set[str]] = {}
        edges: Dict[str, GraphEdge] = {}

        for match in matches:
            if not match.node_id:
                continue
            result = traverser.traverse(
                neighbor_fn=lambda node_id: self.graph_store.neighbors(target_graph, node_id),
                start_ids=[match.node_id],
                max_hops=max_hops,
                node_filter=in_scope,
            )
            for edge in result.edges:
                edges.setdefault(edge.id, edge)

            for node_id, path in result.best_paths.items():
                if path.hops == 0:
                    continue
                current = best_path.get(node_id)
                if current is None or path.weight > current.weight:
                    best_path[node_id] = path
                if path.hops <= 1:
                    direct_tags.setdefault(node_id, set()).add(match.tag_name)

        scored: Dict[str, ScoredNode] = {}
        dampeners: Dict[str, Dict[str, float]] = {}
        for node_id, path in best_path.items():
            node = get_node(node_id)
            if node is None or node.type == NodeType.TAG:
                continue

            hierarchy_decay = self.hopp_decay ** max(0, path.hops - 1)
            persona_relevance = relevance.get(node_id, 1.0)
            matched_tag_count = len(direct_tags.get(node_id, set()))
            overlap_boost = 1.0 + 0.10 * max(0, matched_tag_count - 1)

            raw_score = (
                tag_confidence
                * hierarchy_decay
                * path.weight
                * persona_relevance
                * overlap_boost
            )
            score = round(min(0.98, raw_score), 4)
            confidence = round(
                min(1.0, tag_confidence * hierarchy_decay * path.weight), 4
            )
            scored[node_id] = ScoredNode(
                node=node,
                score=score,
                confidence=confidence,
                retrieval_source="tag_traversal",
            )
            dampeners[node_id] = {
                "hierarchy_decay": hierarchy_decay,
                "path_weight": path.weight,
                "persona_relevance": persona_relevance,
            }

        return scored, best_path, list(edges.values()), dampeners

    # -- helpers ----------------------------------------------------------

    def _resolutions_for(
        self, matches: List[TagMatch], graph_tags: List[GraphNode]
    ) -> List[TagResolution]:
        """Report how each matched tag resolved, including alias provenance."""
        by_title = {t.title.lower().strip(): t for t in graph_tags}
        resolutions: List[TagResolution] = []
        for match in matches:
            tag_node = by_title.get(match.tag_name.lower().strip())
            is_alias = bool(
                tag_node
                and (
                    tag_node.metadata.get("is_alias")
                    or (tag_node.tags and tag_node.tags[0] != tag_node.title)
                )
            )
            canonical = None
            if is_alias and tag_node is not None and tag_node.tags:
                canonical = tag_node.tags[0]
            resolutions.append(
                TagResolution(
                    raw=match.tag_name,
                    canonical=canonical or match.tag_name,
                    confidence=match.score,
                    rule=ResolutionRule.ALIAS if is_alias else ResolutionRule.EXACT,
                )
            )
        return resolutions

    @staticmethod
    def _dedupe(nodes: List[ScoredNode]) -> List[ScoredNode]:
        """
        Collapse nodes that render identically.

        The same heading ingested from two source documents produces two nodes; an
        agent does not benefit from seeing the identical title twice.
        """
        seen: Set[Tuple[str, str]] = set()
        kept: List[ScoredNode] = []
        for scored in nodes:
            key = (scored.node.title.strip().lower(), scored.node.type.value)
            if key in seen:
                continue
            seen.add(key)
            kept.append(scored)
        return kept

    @staticmethod
    def _vector_dampeners(
        node: GraphNode, scope: PersonaScope
    ) -> Dict[str, float]:
        """Vector hits have no path to decay; only perspective and origin dampen them."""
        _, _, relevance = evaluate_node(node, scope)
        return {
            "hierarchy_decay": 1.0,
            "path_weight": 1.0,
            "persona_relevance": relevance,
        }

    def _confidence_for(
        self,
        base_signal: float,
        top_node: Optional[ScoredNode],
        dampeners: Dict[str, Dict[str, float]],
        blocked: Dict[str, ScopeBlock],
    ):
        """Build the tiered confidence verdict for this result."""
        alternatives: List[str] = []
        for block in blocked.values():
            alternatives.extend(block.available_personas)

        if top_node is None:
            return build_breakdown(
                base_signal=0.0,
                floor=self.dampener_floor,
                alternative_personas=alternatives,
            )

        node_dampeners = dampeners.get(top_node.node.id, {})
        return build_breakdown(
            base_signal=base_signal if base_signal > 0 else top_node.score,
            hierarchy_decay=node_dampeners.get("hierarchy_decay", 1.0),
            path_weight=node_dampeners.get("path_weight", 1.0),
            persona_relevance=node_dampeners.get("persona_relevance", 1.0),
            source_confidence=source_confidence(top_node.node),
            floor=self.dampener_floor,
            alternative_personas=alternatives,
        )

    def _detect_coverage_gaps(
        self,
        query: str,
        retrieved_nodes: List[ScoredNode],
        top_k: int,
    ) -> bool:
        """Detect if graph traversal results are insufficient or missing crucial node types."""
        if len(retrieved_nodes) < min(3, top_k):
            return True

        q_lower = query.lower()
        needs_example = any(
            w in q_lower for w in ["example", "sample", "code", "how to", "snippet"]
        )
        has_example = any(n.node.type == NodeType.EXAMPLE for n in retrieved_nodes)
        if needs_example and not has_example:
            return True

        return False

    def _execute_vector_search(
        self,
        query: str,
        target_graph: str,
        exclude_ids: Set[str],
        top_k: int,
        in_scope: Optional[Callable[[str], bool]] = None,
    ) -> List[ScoredNode]:
        """Compute query embedding and retrieve associative nearest neighbors."""
        q_vec = self.embedder.embed_text(query)
        hits = self.vector_store.similarity_search(target_graph, q_vec, top_k=top_k * 2)

        results: List[ScoredNode] = []
        for node_id, sim in hits:
            if node_id in exclude_ids:
                continue
            if in_scope is not None and not in_scope(node_id):
                continue
            node = self.graph_store.get_node(target_graph, node_id)
            if node and node.type != NodeType.TAG:
                results.append(
                    ScoredNode(
                        node=node,
                        score=round(sim, 4),
                        confidence=round(sim * 0.9, 4),
                        retrieval_source="vector_similarity",
                    )
                )
            if len(results) >= top_k:
                break
        return results

    def _build_traversal_chains(
        self, node_ids: List[str], edges: List[GraphEdge]
    ) -> List[List[str]]:
        """Construct multi-hop linear chains from retrieved edges."""
        adj: Dict[str, List[str]] = {}
        for e in edges:
            adj.setdefault(e.source_id, []).append(e.target_id)

        chains: List[List[str]] = []
        for nid in node_ids:
            if nid in adj:
                for target in adj[nid]:
                    chain = [nid, target]
                    if target in adj:
                        for next_target in adj[target]:
                            chain.append(next_target)
                            break
                    chains.append(chain)
        return chains[:5]
