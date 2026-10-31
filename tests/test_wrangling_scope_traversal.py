"""
Tests for perspective scoping and weighted traversal.

Two contracts:
- perspective is a retrieval parameter with a fail-closed default: no declared persona
  never means "show everything"
- a walk multiplies edge weights, prunes weak edges, refuses blocked edges, and never
  loops
"""

from retelldb.models.schema import EdgeType, GraphEdge, GraphNode, NodeType
from retelldb.wrangling.scope import (
    BlockReason,
    PersonaScope,
    evaluate_node,
    filter_nodes,
)
from retelldb.wrangling.traversal import WeightedTraverser


def node(node_id="n1", personas=None, sensitivity="shared", title="Node"):
    return GraphNode(
        id=node_id,
        title=title,
        type=NodeType.CONCEPT,
        personas=personas or [],
        sensitivity=sensitivity,
    )


# -- scope ------------------------------------------------------------------


def test_shared_content_is_visible_without_a_persona():
    visible, reason, relevance = evaluate_node(node(), PersonaScope())
    assert visible is True
    assert reason is None
    assert relevance == 0.85


def test_persona_scoped_content_is_hidden_when_scope_is_unset():
    visible, reason, relevance = evaluate_node(
        node(personas=["student"]), PersonaScope()
    )
    assert visible is False
    assert reason == BlockReason.SCOPE_UNSET
    assert relevance == 0.0


def test_private_content_of_another_persona_is_blocked_before_anything_else():
    visible, reason, _ = evaluate_node(
        node(personas=["student"], sensitivity="private"),
        PersonaScope(persona_id="founder"),
    )
    assert visible is False
    assert reason == BlockReason.PRIVATE_OTHER_PERSONA


def test_persona_scoped_content_is_visible_to_its_persona_at_full_relevance():
    visible, reason, relevance = evaluate_node(
        node(personas=["student"]), PersonaScope(persona_id="student")
    )
    assert visible is True
    assert reason is None
    assert relevance == 1.0


def test_shared_content_can_be_switched_off_for_a_surface():
    visible, reason, _ = evaluate_node(
        node(), PersonaScope(allow_shared=False)
    )
    assert visible is False
    assert reason == BlockReason.SHARED_EXCLUDED


def test_filter_nodes_reports_every_block_with_its_personas():
    kept, blocked, relevance = filter_nodes(
        [node("a"), node("b", personas=["student"], sensitivity="private")],
        PersonaScope(persona_id="founder"),
    )
    assert [n.id for n in kept] == ["a"]
    assert len(blocked) == 1
    assert blocked[0].node_id == "b"
    assert blocked[0].available_personas == ["student"]
    assert relevance["a"] == 0.85


# -- traversal --------------------------------------------------------------


def edge(edge_id, source, target, weight=1.0, edge_type=EdgeType.RELATED, **meta):
    return GraphEdge(
        id=edge_id,
        source_id=source,
        target_id=target,
        type=edge_type,
        weight=weight,
        metadata=meta,
    )


def neighbor_map(mapping):
    """Build a neighbor_fn from {node_id: [(edge, neighbour)]}."""
    return lambda node_id: mapping.get(node_id, [])


def test_chain_weight_is_the_product_along_the_path():
    graph = {
        "tag": [(edge("e1", "tag", "a", 0.8), "a")],
        "a": [(edge("e2", "a", "b", 0.5), "b")],
        "b": [],
    }
    result = WeightedTraverser().traverse(neighbor_map(graph), ["tag"], max_hops=3)

    assert result.best_paths["a"].weight == 0.8
    assert result.best_paths["b"].weight == 0.4
    assert result.best_paths["b"].node_ids == ["tag", "a", "b"]
    assert result.best_paths["b"].describe()["weights"] == [0.8, 0.5]


def test_best_path_wins_when_two_routes_reach_the_same_node():
    graph = {
        "tag": [
            (edge("weak", "tag", "a", 0.3), "a"),
            (edge("strong", "tag", "b", 0.9), "b"),
        ],
        "b": [(edge("hop", "b", "a", 0.9), "a")],
        "a": [],
    }
    result = WeightedTraverser().traverse(neighbor_map(graph), ["tag"], max_hops=3)
    # 0.9 * 0.9 = 0.81 beats the direct 0.3 edge
    assert result.best_paths["a"].weight == 0.81
    assert result.best_paths["a"].node_ids == ["tag", "b", "a"]


def test_blocked_edges_are_never_walked():
    graph = {
        "tag": [(edge("e1", "tag", "a", 0.1, EdgeType.BLOCKS), "a")],
        "a": [],
    }
    result = WeightedTraverser().traverse(neighbor_map(graph), ["tag"], max_hops=3)
    assert "a" not in result.best_paths
    assert "e1" in result.blocked_edges


def test_edge_flagged_as_blocking_in_metadata_is_skipped():
    graph = {"tag": [(edge("e1", "tag", "a", 0.9, blocks=True), "a")], "a": []}
    result = WeightedTraverser().traverse(neighbor_map(graph), ["tag"], max_hops=3)
    assert "a" not in result.best_paths


def test_edges_below_the_floor_are_pruned():
    graph = {"tag": [(edge("e1", "tag", "a", 0.01), "a")], "a": []}
    result = WeightedTraverser(min_edge_weight=0.05).traverse(
        neighbor_map(graph), ["tag"], max_hops=3
    )
    assert "a" not in result.best_paths
    assert "e1" in result.pruned_edges


def test_cycles_terminate():
    graph = {
        "tag": [(edge("e1", "tag", "a", 1.0), "a")],
        "a": [(edge("e2", "a", "tag", 1.0), "tag")],
    }
    result = WeightedTraverser().traverse(neighbor_map(graph), ["tag"], max_hops=5)
    assert result.best_paths["a"].node_ids == ["tag", "a"]
    for path in result.paths:
        assert len(path.node_ids) == len(set(path.node_ids))


def test_hop_budget_is_enforced():
    graph = {
        "tag": [(edge("e1", "tag", "a", 1.0), "a")],
        "a": [(edge("e2", "a", "b", 1.0), "b")],
        "b": [(edge("e3", "b", "c", 1.0), "c")],
        "c": [],
    }
    result = WeightedTraverser().traverse(neighbor_map(graph), ["tag"], max_hops=2)
    assert "b" in result.best_paths
    assert "c" not in result.best_paths


def test_node_filter_blocks_traversal_through_a_node():
    graph = {
        "tag": [(edge("e1", "tag", "a", 1.0), "a")],
        "a": [(edge("e2", "a", "b", 1.0), "b")],
        "b": [],
    }
    result = WeightedTraverser().traverse(
        neighbor_map(graph), ["tag"], max_hops=3, node_filter=lambda nid: nid != "a"
    )
    assert "a" not in result.best_paths
    assert "b" not in result.best_paths
