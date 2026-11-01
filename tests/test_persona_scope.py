"""
the connection + scope rules, pinned.

runs two ways: `python3 retelldb/tests/test_scope.py` (standalone) or
`pytest retelldb/tests/` when pytest is around. plain asserts, no deps.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from retelldb.models.connections import (  # noqa: E402
    Connection,
    example_connections,
    normalize_handle,
    platform_class,
    resolve_connection,
)
from retelldb.routing.scope import (  # noqa: E402
    CONNECTION_TAG_PREFIX,
    MemoryScope,
    carry_decision,
    filter_nodes,
    node_connection,
    provenance,
)


# --- handles ---------------------------------------------------------


def test_platform_classes():
    assert platform_class("zoom") == "meeting"
    assert platform_class("google_meet") == "meeting"
    assert platform_class("discord") == "private"
    assert platform_class("whatsapp") == "private"
    assert platform_class("email") == "private"
    assert platform_class("twitch") == "public_social"
    assert platform_class("youtube") == "public_social"
    assert platform_class("myspace") == "unknown"
    assert platform_class(None) == "unknown"


def test_handle_normalization():
    assert normalize_handle("email", " Priya@Example.COM ") == "priya@example.com"
    assert normalize_handle("zoom", "PRIYA@example.com") == "priya@example.com"
    assert normalize_handle("discord", "Priya#1234") == "priya#1234"
    assert normalize_handle("whatsapp", "1 555 000 0000") == "+15550000000"
    assert normalize_handle("telegram", "+1-555-000-0000") == "+15550000000"
    assert normalize_handle("email", "   ") == ""


def test_resolution_anchors_one_person_across_surfaces():
    people = example_connections()
    by_discord = resolve_connection(people, "discord", "PRIYA-ID-PLACEHOLDER")
    by_email = resolve_connection(people, "email", "priya@example.com")
    by_zoom = resolve_connection(people, "zoom", "Priya@Example.com")
    assert by_discord is not None and by_email is not None and by_zoom is not None
    # all three resolves land on ONE connection: the continuation rule
    assert by_discord.id == by_email.id == by_zoom.id == "priya"


def test_resolution_is_honest_when_ambiguous_or_unknown():
    a = Connection(id="a", label="a", handles={"email": ["shared@example.com"]})
    b = Connection(id="b", label="b", handles={"email": ["shared@example.com"]})
    assert resolve_connection([a, b], "email", "shared@example.com") is None
    assert resolve_connection([a], "email", "nobody@example.com") is None
    assert resolve_connection([a], "email", "") is None


# --- carry decisions -------------------------------------------------


def test_same_person_cross_platform_carries():
    src = MemoryScope(persona_id="p", platform="zoom", connection_id="priya")
    dst = MemoryScope(persona_id="p", platform="discord", connection_id="priya")
    tags = ["platform:zoom", "connection:priya"]
    d = carry_decision(src, dst, tags)
    assert d.carry is True


def test_other_persons_context_never_crosses():
    src = MemoryScope(persona_id="p", platform="zoom", connection_id="priya")
    dst = MemoryScope(persona_id="p", platform="discord", connection_id="rahul")
    d = carry_decision(src, dst, ["platform:zoom", "connection:priya"])
    assert d.carry is False
    assert "another person" in d.reason


def test_anchored_context_does_not_surface_unanchored():
    src = MemoryScope(persona_id="p", platform="zoom", connection_id="priya")
    dst = MemoryScope(persona_id="p", platform="zoom", connection_id=None)
    d = carry_decision(src, dst, ["platform:zoom", "connection:priya"])
    assert d.carry is False
    assert "unanchored" in d.reason


def test_meeting_captures_never_surface_on_public_surfaces():
    src = MemoryScope(persona_id="p", platform="zoom")
    dst = MemoryScope(persona_id="p", platform="twitch")
    d = carry_decision(src, dst, ["platform:zoom"])
    assert d.carry is False
    assert "public surfaces" in d.reason
    # the same meeting memory into a private dm still carries
    ok = carry_decision(src, MemoryScope(persona_id="p", platform="discord"), ["platform:zoom"])
    assert ok.carry is True


def test_another_personas_memory_drops_unless_shared():
    src = MemoryScope(persona_id="founder", platform="email")
    dst = MemoryScope(persona_id="student", platform="email")
    assert carry_decision(src, dst, ["platform:email"]).carry is False
    # the shared graph carries across personas
    assert carry_decision(src, dst, ["platform:email"], shared=True).carry is True


def test_private_sensitivity_stays_in_private_with_its_person():
    tags = ["platform:zoom", "connection:priya", "sensitivity:private"]
    src = MemoryScope(persona_id="p", platform="zoom", connection_id="priya")
    # private -> public_social is already stopped by rule 5, and by rule 6
    d1 = carry_decision(src, MemoryScope(persona_id="p", platform="twitch", connection_id="priya"), tags)
    assert d1.carry is False
    # private -> private with the same person carries
    d2 = carry_decision(src, MemoryScope(persona_id="p", platform="whatsapp", connection_id="priya"), tags)
    assert d2.carry is True
    # private -> private but a different person drops
    d3 = carry_decision(src, MemoryScope(persona_id="p", platform="whatsapp", connection_id="rahul"), tags)
    assert d3.carry is False


def test_unknown_surface_drops_everything():
    src = MemoryScope(persona_id="p", platform="zoom")
    dst = MemoryScope(persona_id="p", platform="myspace")
    d = carry_decision(src, dst, ["platform:zoom"])
    assert d.carry is False
    assert "unknown surface" in d.reason


def test_global_memory_carries_everywhere_modeled():
    src = MemoryScope(persona_id="p", platform="email")
    dst = MemoryScope(persona_id="p", platform="zoom", connection_id="priya")
    d = carry_decision(src, dst, [])  # no platform or connection tag = global
    assert d.carry is True


# --- filtering + provenance -----------------------------------------


class _Node:
    def __init__(self, id, title, tags, namespace="p"):
        self.id = id
        self.title = title
        self.tags = tags
        self.namespace = namespace


class _Scored:
    def __init__(self, node):
        self.node = node


def test_filter_nodes_keeps_and_reports_each_drop_with_a_reason():
    src = MemoryScope(persona_id="p", platform="zoom")
    dst = MemoryScope(persona_id="p", platform="discord", connection_id="priya")
    nodes = [
        _Scored(_Node("1", "priya chat", ["platform:zoom", "connection:priya"])),
        _Scored(_Node("2", "rahul chat", ["platform:zoom", "connection:rahul"])),
        _Scored(_Node("3", "global fact", [])),
    ]
    kept, dropped = filter_nodes(nodes, src, dst, shared_ids=set())
    assert [sn.node.id for sn in kept] == ["1", "3"]
    assert [d["node_id"] for d in dropped] == ["2"]
    assert dropped[0]["reason"]
    assert dropped[0]["node_connection"] == "rahul"


def test_provenance_stamps_are_readable_back():
    scope = MemoryScope(persona_id="p", platform="Discord", connection_id="priya")
    prov = provenance(scope)
    tags = prov["tags"]
    assert "platform:discord" in tags
    assert f"{CONNECTION_TAG_PREFIX}priya" in tags
    assert node_connection(tags) == "priya"


# --- engine wiring ---------------------------------------------------


def test_engine_applies_scope_and_reports_drops():
    """retrieve() with a platform keeps the same person's memory and
    reports what it held back, with reasons. no platform = pre-scope."""
    from retelldb.models.schema import GraphNode, NodeType
    from retelldb.models.query import QueryRequest
    from retelldb.routing.engine import RetrievalEngine

    nodes = {
        "n-priya": GraphNode(
            id="n-priya", title="priya's brief", type=NodeType.FACT,
            namespace="p", tags=["platform:zoom", "connection:priya"],
        ),
        "n-rahul": GraphNode(
            id="n-rahul", title="rahul's aside", type=NodeType.FACT,
            namespace="p", tags=["platform:zoom", "connection:rahul"],
        ),
        "n-global": GraphNode(
            id="n-global", title="a standing preference", type=NodeType.FACT,
            namespace="p", tags=[],
        ),
    }

    class FakeGraphStore:
        def get_tag_nodes(self, graph_name):
            return []

        def get_node(self, graph_name, node_id):
            return nodes.get(node_id)

        def get_subgraph_between(self, graph_name, ids):
            return []

    class FakeVectorStore:
        def __init__(self):
            self.order = ["n-priya", "n-rahul", "n-global"]

        def similarity_search(self, graph_name, query_vec, top_k=10):
            return [(nid, 0.9) for nid in self.order]

    class FakeEmbedder:
        def embed_text(self, text):
            return [0.0]

    engine = RetrievalEngine(FakeGraphStore(), FakeVectorStore())
    engine.embedder = FakeEmbedder()

    scoped = engine.retrieve(QueryRequest(
        query="what did we settle on?",
        persona_id="p",
        platform="discord",
        connection_id="priya",
        include_edges=False,
    ))
    kept = {sn.node.id for sn in scoped.nodes}
    assert kept == {"n-priya", "n-global"}, kept
    assert [d["node_id"] for d in scoped.dropped] == ["n-rahul"]
    assert scoped.dropped[0]["reason"]

    unscoped = engine.retrieve(QueryRequest(query="what did we settle on?", persona_id="p"))
    assert {sn.node.id for sn in unscoped.nodes} == {"n-priya", "n-rahul", "n-global"}
    assert unscoped.dropped == []


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
