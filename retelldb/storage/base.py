"""
Base storage interfaces for graph and vector layers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from retelldb.models.schema import GraphNode, GraphEdge


class BaseGraphStore(ABC):
    """Abstract interface for property graph backends (FalkorDB / In-Memory)."""

    @abstractmethod
    def add_node(self, graph_name: str, node: GraphNode) -> None:
        """Add or update a node in the specified graph."""
        pass

    @abstractmethod
    def add_edge(self, graph_name: str, edge: GraphEdge) -> None:
        """Add an edge between two existing nodes in the graph."""
        pass

    @abstractmethod
    def get_node(self, graph_name: str, node_id: str) -> Optional[GraphNode]:
        """Fetch a node by its ID."""
        pass

    @abstractmethod
    def get_edges(self, graph_name: str, node_id: str) -> List[GraphEdge]:
        """Fetch all connected edges for a given node."""
        pass

    @abstractmethod
    def get_tag_nodes(self, graph_name: str) -> List[GraphNode]:
        """Retrieve all first-class semantic tag nodes in the graph."""
        pass

    @abstractmethod
    def tag_traversal(
        self, 
        graph_name: str, 
        tag_names: List[str], 
        max_hops: int = 3
    ) -> Tuple[List[Tuple[GraphNode, int]], List[GraphEdge]]:
        """
        Traverse the graph from matching tag nodes outward up to max_hops.
        Returns: (List of (GraphNode, hop_distance), List of GraphEdge)
        """
        pass

    @abstractmethod
    def get_subgraph_between(
        self, 
        graph_name: str, 
        node_ids: List[str]
    ) -> List[GraphEdge]:
        """Find connecting edges between a subset of nodes."""
        pass

    @abstractmethod
    def get_nodes_by_source(
        self, 
        graph_name: str, 
        source_uri: str
    ) -> List[GraphNode]:
        """Find all nodes that originated from a specific source URI."""
        pass

    @abstractmethod
    def delete_nodes(
        self, 
        graph_name: str, 
        node_ids: List[str]
    ) -> None:
        """Delete nodes and all their incident edges from the graph."""
        pass

    @abstractmethod
    def get_status(self, graph_name: str) -> Dict[str, Any]:
        """Return metrics (node count, edge count, tag count) for the graph."""
        pass


class BaseVectorStore(ABC):
    """Abstract interface for vector similarity backends (pgvector / In-Memory)."""

    @abstractmethod
    def add_embedding(
        self, 
        graph_name: str, 
        node_id: str, 
        embedding: List[float], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a contextual node embedding."""
        pass

    @abstractmethod
    def similarity_search(
        self, 
        graph_name: str, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return top_k (node_id, similarity_score) pairs in descending order."""
        pass

    @abstractmethod
    def delete_embeddings(
        self, 
        graph_name: str, 
        node_ids: List[str]
    ) -> None:
        """Remove embeddings for specific node IDs."""
        pass

    @abstractmethod
    def get_status(self, graph_name: str) -> Dict[str, Any]:
        """Return vector store stats."""
        pass
