"""
Weighted graph traversal.

Retrieval in retelldb is a walk, not a nearest-neighbour scan. This module owns the
walk so that both storage backends share one scoring implementation:

- every edge contributes its explicit weight, and chain weight is the product
  along the path (with a floor, so a long weak chain cannot round to zero)
- a minimum edge weight prunes noise before the hop budget is spent
- `blocks` edges are never walked through: they are negative associations
- cycles cannot loop: a path never revisits a node, and the best path per node wins

The traversal returns the paths themselves, not just the winning nodes, so an agent
can be shown why each node was pulled in.
"""

import heapq
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from retelldb.models.schema import EdgeType, GraphEdge, GraphNode

logger = logging.getLogger("retelldb.wrangling.traversal")

# Node id -> list of (edge, neighbour id)
NeighborFn = Callable[[str], Sequence[Tuple[GraphEdge, str]]]

DEFAULT_MIN_EDGE_WEIGHT = 0.05
DEFAULT_MAX_EXPANSIONS = 500

# Edge types that never carry retrieval signal forward
BLOCKING_EDGE_TYPES = {EdgeType.BLOCKS}


class TraversedPath(BaseModel):
    """One auditable walk from a start tag node to a fetched node."""

    start_id: str
    node_ids: List[str] = Field(default_factory=list)
    edge_ids: List[str] = Field(default_factory=list)
    edge_weights: List[float] = Field(default_factory=list)
    weight: float = 1.0
    hops: int = 0

    def describe(self) -> Dict[str, object]:
        """Serializable trace: nodes, edges and the weight that pulled each edge in."""
        return {
            "nodes": list(self.node_ids),
            "edges": list(self.edge_ids),
            "weights": list(self.edge_weights),
            "chain_weight": round(self.weight, 6),
            "hops": self.hops,
        }


class TraversalResult(BaseModel):
    """The outcome of a weighted walk from a set of start nodes."""

    paths: List[TraversedPath] = Field(default_factory=list)
    # node id -> the best path that reached it
    best_paths: Dict[str, TraversedPath] = Field(default_factory=dict)
    edges: List[GraphEdge] = Field(default_factory=list)
    blocked_edges: List[str] = Field(default_factory=list)
    pruned_edges: List[str] = Field(default_factory=list)
    expansions: int = 0

    def nodes_ranked(self) -> List[Tuple[str, TraversedPath]]:
        """Node ids ordered by chain weight (descending), then hops, then id."""
        items = list(self.best_paths.items())
        items.sort(key=lambda kv: (-kv[1].weight, kv[1].hops, kv[0]))
        return items


class WeightedTraverser:
    """Best-first traversal over explicit edge weights."""

    def __init__(
        self,
        min_edge_weight: float = DEFAULT_MIN_EDGE_WEIGHT,
        max_expansions: int = DEFAULT_MAX_EXPANSIONS,
        blocking_edge_types: Optional[Sequence[EdgeType]] = None,
    ):
        self.min_edge_weight = min_edge_weight
        self.max_expansions = max_expansions
        self.blocking_edge_types = set(blocking_edge_types or BLOCKING_EDGE_TYPES)

    def _edge_weight(self, edge: GraphEdge) -> Optional[float]:
        """Return the weight to travel this edge, or None if it must not be travelled."""
        if edge.type in self.blocking_edge_types:
            return None
        if bool(edge.metadata.get("blocks")):
            return None
        weight = float(edge.weight if edge.weight is not None else 1.0)
        if weight < self.min_edge_weight:
            return None
        # weights above 1.0 would break the monotone decrease that keeps the walk finite
        return min(weight, 1.0)

    def traverse(
        self,
        neighbor_fn: NeighborFn,
        start_ids: Sequence[str],
        max_hops: int = 3,
        node_filter: Optional[Callable[[str], bool]] = None,
    ) -> TraversalResult:
        """
        Walk outward from `start_ids`.

        `node_filter` is consulted before a node is added: scope decisions happen
        inside the walk so blocked nodes are never traversed through.
        """
        result = TraversalResult()
        if max_hops <= 0 or not start_ids:
            return result

        seen_edges: Dict[str, GraphEdge] = {}
        best: Dict[str, TraversedPath] = {}
        # heap entries: (-weight, hops, node_id, path)
        heap: List[Tuple[float, int, str, TraversedPath]] = []

        for start in start_ids:
            path = TraversedPath(start_id=start, node_ids=[start], weight=1.0, hops=0)
            best[start] = path
            heap.append((-1.0, 0, start, path))
        heapq.heapify(heap)

        while heap and result.expansions < self.max_expansions:
            neg_weight, hops, node_id, path = heapq.heappop(heap)
            if hops >= max_hops:
                continue
            result.expansions += 1

            for edge, neighbor_id in neighbor_fn(node_id):
                seen_edges.setdefault(edge.id, edge)
                edge_weight = self._edge_weight(edge)
                if edge_weight is None:
                    marker = result.blocked_edges
                    if edge.type not in self.blocking_edge_types:
                        marker = result.pruned_edges
                    if edge.id not in marker:
                        marker.append(edge.id)
                    continue

                if neighbor_id in path.node_ids:
                    continue  # cycle: a path never revisits a node
                if node_filter is not None and not node_filter(neighbor_id):
                    continue

                chain_weight = path.weight * edge_weight
                if chain_weight < self.min_edge_weight:
                    if edge.id not in result.pruned_edges:
                        result.pruned_edges.append(edge.id)
                    continue

                existing = best.get(neighbor_id)
                if existing is not None and existing.weight >= chain_weight:
                    continue

                next_path = TraversedPath(
                    start_id=path.start_id,
                    node_ids=path.node_ids + [neighbor_id],
                    edge_ids=path.edge_ids + [edge.id],
                    edge_weights=path.edge_weights + [round(edge_weight, 6)],
                    weight=chain_weight,
                    hops=hops + 1,
                )
                best[neighbor_id] = next_path
                heapq.heappush(heap, (-chain_weight, hops + 1, neighbor_id, next_path))

        result.best_paths = best
        result.paths = [p for p in best.values() if p.hops > 0]
        result.edges = list(seen_edges.values())
        return result
