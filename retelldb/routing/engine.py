"""
RetellDB Routing Engine.
Orchestrates audience classification, query tag detection, FalkorDB multi-hop traversal,
pgvector associative vector fallback, gap detection, and subgraph ranking.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Set
from retelldb.models.schema import (
    NodeType,
    AudienceType,
    GraphNode,
    GraphEdge,
)
from retelldb.models.query import (
    QueryRequest,
    QueryStrategy,
    ScoredNode,
    SubGraphResult,
)
from retelldb.storage.base import BaseGraphStore, BaseVectorStore
from retelldb.storage.embedder import ContextualEmbedder
from retelldb.routing.audience import AudienceClassifier
from retelldb.routing.classifier import QueryClassifier
from retelldb.config import RetellDBConfig, default_config

logger = logging.getLogger("retelldb.routing")


class RetrievalEngine:
    """The central query routing and retrieval engine of RetellDB."""

    def __init__(
        self,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        config: Optional[RetellDBConfig] = None
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.config = config or default_config

        self.audience_classifier = AudienceClassifier(self.config.inference)
        self.query_classifier = QueryClassifier()
        self.embedder = ContextualEmbedder(self.config.inference)

    def retrieve(self, request: QueryRequest) -> SubGraphResult:
        """Execute full routing and retrieval lifecycle for an agent query."""
        t_start = time.perf_counter()

        # 1. Determine target graph via audience classifier
        audience = self.audience_classifier.classify(request.query, request.audience)
        target_graph = (
            self.config.storage.dev_graph_name 
            if audience == AudienceType.DEVELOPER 
            else self.config.storage.user_graph_name
        )

        # 2. Check for strong tag signal
        graph_tags = self.graph_store.get_tag_nodes(target_graph)
        has_signal, matched_tags, tag_confidence = self.query_classifier.evaluate_tag_signal(
            query=request.query,
            graph_tags=graph_tags,
            threshold=request.tag_confidence_threshold
        )

        retrieved_nodes_map: Dict[str, ScoredNode] = {}
        traversal_edges: List[GraphEdge] = []
        strategy: QueryStrategy

        if has_signal:
            # 3A. Semantic Tag Traversal — run per matched tag to track origin
            # Accumulate per-node: best hop distance and set of originating tags
            node_best_hop: Dict[str, int] = {}
            node_tag_origins: Dict[str, Set[str]] = {}
            node_ref: Dict[str, Any] = {}  # node_id -> GraphNode

            for tag_name in matched_tags:
                t_nodes, t_edges = self.graph_store.tag_traversal(
                    graph_name=target_graph,
                    tag_names=[tag_name],
                    max_hops=request.max_hops
                )
                traversal_edges.extend(t_edges)

                for item in t_nodes:
                    if isinstance(item, tuple):
                        n, hop = item
                    else:
                        n, hop = item, 1

                    if n.type != NodeType.TAG:
                        nid = n.id
                        node_ref[nid] = n
                        # Track best (minimum) hop from any tag
                        if nid not in node_best_hop or hop < node_best_hop[nid]:
                            node_best_hop[nid] = hop
                        # Only count direct tag membership (hop 1) for the tag boost.
                        # Nodes at hop 2+ are merely graph neighbors, not directly tagged.
                        if hop <= 1:
                            node_tag_origins.setdefault(nid, set()).add(tag_name)

            # Deduplicate traversal edges
            traversal_edges = list({e.id: e for e in traversal_edges}.values())

            # Score nodes: weight by hop-distance decay AND number of matched tags
            for nid, n in node_ref.items():
                hop = node_best_hop[nid]
                n_tags_matched = len(node_tag_origins.get(nid, set()))
                effective_hop = max(1, hop)
                decay = 0.85 ** (effective_hop - 1)
                # Boost for multi-tag matches: each additional tag adds 0.10
                tag_boost = 1.0 + 0.10 * (n_tags_matched - 1)
                raw_score = tag_confidence * decay * tag_boost
                score = round(min(0.98, raw_score), 4)
                conf = round(tag_confidence * decay, 4)
                retrieved_nodes_map[nid] = ScoredNode(
                    node=n,
                    score=score,
                    confidence=conf,
                    retrieval_source="tag_traversal"
                )

            # 3B. Coverage Gap Check
            has_gaps = self._detect_coverage_gaps(
                query=request.query,
                retrieved_nodes=list(retrieved_nodes_map.values()),
                top_k=request.top_k
            )

            if has_gaps:
                strategy = QueryStrategy.HYBRID
                gap_nodes = self._execute_vector_search(
                    query=request.query,
                    target_graph=target_graph,
                    exclude_ids=set(retrieved_nodes_map.keys()),
                    top_k=request.top_k - len(retrieved_nodes_map) if len(retrieved_nodes_map) < request.top_k else 2
                )
                for gn in gap_nodes:
                    gn.retrieval_source = "gap_fill"
                    retrieved_nodes_map[gn.node.id] = gn
            else:
                strategy = QueryStrategy.TAG_TRAVERSAL

        else:
            # 3C. Latent Vector Similarity Fallback on Vector Store
            strategy = QueryStrategy.VECTOR_FALLBACK
            vector_nodes = self._execute_vector_search(
                query=request.query,
                target_graph=target_graph,
                exclude_ids=set(),
                top_k=request.top_k
            )
            for vn in vector_nodes:
                retrieved_nodes_map[vn.node.id] = vn

        # 4. Rank nodes
        scored_nodes = list(retrieved_nodes_map.values())
        scored_nodes.sort(key=lambda x: (x.score, x.confidence), reverse=True)
        top_nodes = scored_nodes[:request.top_k]

        # 5. Connect edges in retrieved subgraph
        final_edges: List[GraphEdge] = []
        traversal_chains: List[List[str]] = []

        if request.include_edges and top_nodes:
            top_ids = [sn.node.id for sn in top_nodes]
            # Fetch internal edges between retrieved nodes
            subgraph_edges = self.graph_store.get_subgraph_between(target_graph, top_ids)
            edge_map = {e.id: e for e in (subgraph_edges + traversal_edges)}
            # Keep edges where at least one end is in top_nodes
            top_set = set(top_ids)
            final_edges = [
                e for e in edge_map.values()
                if e.source_id in top_set or e.target_id in top_set
            ]
            traversal_chains = self._build_traversal_chains(top_ids, final_edges)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return SubGraphResult(
            query=request.query,
            target_graph=target_graph,
            retrieval_path=strategy,
            nodes=top_nodes,
            edges=final_edges,
            traversal_chains=traversal_chains,
            confidence_signals={
                "tag_confidence": tag_confidence,
                "matched_tags_count": float(len(matched_tags)),
                "avg_node_score": (
                    sum(n.score for n in top_nodes) / len(top_nodes)
                    if top_nodes else 0.0
                ),
            },
            execution_time_ms=round(elapsed_ms, 2)
        )

    def _detect_coverage_gaps(
        self, 
        query: str, 
        retrieved_nodes: List[ScoredNode], 
        top_k: int
    ) -> bool:
        """Detect if graph traversal results are insufficient or missing crucial node types."""
        if len(retrieved_nodes) < min(3, top_k):
            return True

        # Check if query requests code/examples but none returned
        q_lower = query.lower()
        needs_example = any(w in q_lower for w in ["example", "sample", "code", "how to", "snippet"])
        has_example = any(n.node.type == NodeType.EXAMPLE for n in retrieved_nodes)
        if needs_example and not has_example:
            return True

        return False

    def _execute_vector_search(
        self,
        query: str,
        target_graph: str,
        exclude_ids: Set[str],
        top_k: int
    ) -> List[ScoredNode]:
        """Compute query embedding and retrieve associative nearest neighbors."""
        q_vec = self.embedder.embed_text(query)
        hits = self.vector_store.similarity_search(target_graph, q_vec, top_k=top_k * 2)

        results: List[ScoredNode] = []
        for node_id, sim in hits:
            if node_id in exclude_ids:
                continue
            node = self.graph_store.get_node(target_graph, node_id)
            if node and node.type != NodeType.TAG:
                results.append(
                    ScoredNode(
                        node=node,
                        score=round(sim, 4),
                        confidence=round(sim * 0.9, 4),
                        retrieval_source="vector_similarity"
                    )
                )
            if len(results) >= top_k:
                break
        return results

    def _build_traversal_chains(
        self, 
        node_ids: List[str], 
        edges: List[GraphEdge]
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
