"""
In-memory implementation of property graph and vector stores.
Provides full multi-hop traversal with hop distance tracking and cosine vector similarity.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque
import math
from retelldb.models.schema import GraphNode, GraphEdge, NodeType, EdgeType
from retelldb.storage.base import BaseGraphStore, BaseVectorStore


class MemoryGraphStore(BaseGraphStore):
    """In-memory property graph store with multi-hop tag traversal."""

    def __init__(self):
        # graph_name -> {"nodes": {id: node}, "edges": {id: edge}, "out_adj": {id: [edge]}, "in_adj": {id: [edge]}}
        self._graphs: Dict[str, Dict[str, Any]] = {}

    def _ensure_graph(self, graph_name: str) -> Dict[str, Any]:
        if graph_name not in self._graphs:
            self._graphs[graph_name] = {
                "nodes": {},
                "edges": {},
                "out_adj": {},
                "in_adj": {},
            }
        return self._graphs[graph_name]

    def add_node(self, graph_name: str, node: GraphNode) -> None:
        g = self._ensure_graph(graph_name)
        g["nodes"][node.id] = node
        if node.id not in g["out_adj"]:
            g["out_adj"][node.id] = []
        if node.id not in g["in_adj"]:
            g["in_adj"][node.id] = []

    def add_edge(self, graph_name: str, edge: GraphEdge) -> None:
        g = self._ensure_graph(graph_name)
        g["edges"][edge.id] = edge
        g["out_adj"].setdefault(edge.source_id, []).append(edge)
        g["in_adj"].setdefault(edge.target_id, []).append(edge)

    def get_node(self, graph_name: str, node_id: str) -> Optional[GraphNode]:
        g = self._ensure_graph(graph_name)
        return g["nodes"].get(node_id)

    def get_edges(self, graph_name: str, node_id: str) -> List[GraphEdge]:
        g = self._ensure_graph(graph_name)
        out_edges = g["out_adj"].get(node_id, [])
        in_edges = g["in_adj"].get(node_id, [])
        return list({e.id: e for e in out_edges + in_edges}.values())

    def get_tag_nodes(self, graph_name: str) -> List[GraphNode]:
        g = self._ensure_graph(graph_name)
        return [n for n in g["nodes"].values() if n.type == NodeType.TAG]

    def tag_traversal(
        self, 
        graph_name: str, 
        tag_names: List[str], 
        max_hops: int = 3
    ) -> Tuple[List[Tuple[GraphNode, int]], List[GraphEdge]]:
        g = self._ensure_graph(graph_name)
        matched_tag_nodes: List[GraphNode] = []
        normalized_query_tags = {t.lower().strip() for t in tag_names}

        # Find matching tag nodes
        for node in g["nodes"].values():
            if node.type == NodeType.TAG and node.title.lower().strip() in normalized_query_tags:
                matched_tag_nodes.append(node)

        if not matched_tag_nodes:
            return [], []

        node_hops: Dict[str, int] = {}
        collected_edges: Dict[str, GraphEdge] = {}
        queue: deque = deque()  # (node_id, current_hop)

        for tag_node in matched_tag_nodes:
            node_hops[tag_node.id] = 0
            queue.append((tag_node.id, 0))

        # BFS Traversal
        while queue:
            curr_id, hop = queue.popleft()
            if hop >= max_hops:
                continue

            # Check neighbors
            neighbors = g["out_adj"].get(curr_id, []) + g["in_adj"].get(curr_id, [])
            for edge in neighbors:
                other_id = edge.target_id if edge.source_id == curr_id else edge.source_id
                collected_edges[edge.id] = edge
                if other_id not in node_hops or node_hops[other_id] > hop + 1:
                    node_hops[other_id] = hop + 1
                    queue.append((other_id, hop + 1))

        retrieved_nodes_with_hops = [
            (g["nodes"][nid], hop)
            for nid, hop in node_hops.items()
            if nid in g["nodes"]
        ]
        return retrieved_nodes_with_hops, list(collected_edges.values())

    def get_subgraph_between(
        self, 
        graph_name: str, 
        node_ids: List[str]
    ) -> List[GraphEdge]:
        g = self._ensure_graph(graph_name)
        id_set = set(node_ids)
        edges = []
        for edge in g["edges"].values():
            if edge.source_id in id_set and edge.target_id in id_set:
                edges.append(edge)
        return edges

    def get_nodes_by_source(self, graph_name: str, source_uri: str) -> List[GraphNode]:
        g = self._ensure_graph(graph_name)
        norm_source = source_uri.strip().lower()
        return [
            n for n in g["nodes"].values()
            if n.source_uri and n.source_uri.strip().lower() == norm_source
        ]

    def delete_nodes(self, graph_name: str, node_ids: List[str]) -> None:
        g = self._ensure_graph(graph_name)
        id_set = set(node_ids)
        
        # Delete nodes
        for nid in id_set:
            g["nodes"].pop(nid, None)

        # Delete incident edges
        edges_to_delete = [
            e_id for e_id, e in g["edges"].items()
            if e.source_id in id_set or e.target_id in id_set
        ]
        for e_id in edges_to_delete:
            edge = g["edges"].pop(e_id, None)
            if edge:
                if edge.source_id in g["out_adj"]:
                    g["out_adj"][edge.source_id] = [
                        x for x in g["out_adj"][edge.source_id] if x.id != e_id
                    ]
                if edge.target_id in g["in_adj"]:
                    g["in_adj"][edge.target_id] = [
                        x for x in g["in_adj"][edge.target_id] if x.id != e_id
                    ]

    def get_status(self, graph_name: str) -> Dict[str, Any]:
        g = self._ensure_graph(graph_name)
        nodes = g["nodes"]
        edges = g["edges"]
        tag_count = sum(1 for n in nodes.values() if n.type == NodeType.TAG)
        return {
            "backend": "memory",
            "graph_name": graph_name,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_tags": tag_count,
            "node_types": {
                t.value: sum(1 for n in nodes.values() if n.type == t)
                for t in NodeType
            }
        }


class MemoryVectorStore(BaseVectorStore):
    """In-memory vector store with cosine similarity."""

    def __init__(self):
        # graph_name -> {node_id: {"embedding": [...], "metadata": {...}}}
        self._vectors: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _ensure_graph(self, graph_name: str) -> Dict[str, Dict[str, Any]]:
        if graph_name not in self._vectors:
            self._vectors[graph_name] = {}
        return self._vectors[graph_name]

    def add_embedding(
        self, 
        graph_name: str, 
        node_id: str, 
        embedding: List[float], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        g = self._ensure_graph(graph_name)
        norm = math.sqrt(sum(x * x for x in embedding)) or 1.0
        normalized = [x / norm for x in embedding]
        g[node_id] = {
            "embedding": normalized,
            "metadata": metadata or {}
        }

    def similarity_search(
        self, 
        graph_name: str, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        g = self._ensure_graph(graph_name)
        if not g or not query_embedding:
            return []

        q_norm = math.sqrt(sum(x * x for x in query_embedding)) or 1.0
        q_normed = [x / q_norm for x in query_embedding]

        scores = []
        for node_id, data in g.items():
            vec = data["embedding"]
            dot = sum(a * b for a, b in zip(q_normed, vec))
            sim = max(0.0, min(1.0, (dot + 1.0) / 2.0))
            scores.append((node_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def delete_embeddings(self, graph_name: str, node_ids: List[str]) -> None:
        g = self._ensure_graph(graph_name)
        for nid in node_ids:
            g.pop(nid, None)

    def get_status(self, graph_name: str) -> Dict[str, Any]:
        g = self._ensure_graph(graph_name)
        return {
            "backend": "memory",
            "graph_name": graph_name,
            "total_vectors": len(g)
        }
