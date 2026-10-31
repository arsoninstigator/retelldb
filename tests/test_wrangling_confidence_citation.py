"""
Tests for confidence tiering, source citations, and the end-to-end wrangler behaviour.
"""

from retelldb.client import RetellDB
from retelldb.models.schema import GraphNode, NodeType, SourceRef
from retelldb.wrangling.citation import SourceVerifier, citations_for_node
from retelldb.wrangling.confidence import (
    ConfidenceTier,
    build_breakdown,
    tier_for,
)
from retelldb.wrangling.scope import BlockReason
from retelldb.wrangling.taxonomy import TagTaxonomy


# -- confidence -------------------------------------------------------------


def test_tier_boundaries_match_the_documented_bands():
    assert tier_for(0.9) == ConfidenceTier.HIGH
    assert tier_for(0.75) == ConfidenceTier.HIGH
    assert tier_for(0.74) == ConfidenceTier.UNCERTAIN
    assert tier_for(0.45) == ConfidenceTier.UNCERTAIN
    assert tier_for(0.44) == ConfidenceTier.NEEDS_CLARIFICATION
    assert tier_for(0.25) == ConfidenceTier.NEEDS_CLARIFICATION
    assert tier_for(0.24) == ConfidenceTier.UNAVAILABLE


def test_clean_signals_land_high_and_say_answer_directly():
    breakdown = build_breakdown(base_signal=1.0)
    assert breakdown.tier == ConfidenceTier.HIGH
    assert breakdown.total == 1.0
    assert breakdown.dominant_dampener is None
    assert breakdown.next_actions == ["answer_directly"]
    assert breakdown.is_answerable() is True


def test_one_weak_component_hedges_instead_of_answering_boldly():
    breakdown = build_breakdown(
        base_signal=0.8, hierarchy_decay=0.85, persona_relevance=0.5, path_weight=1.0
    )
    assert breakdown.tier == ConfidenceTier.UNCERTAIN
    assert breakdown.dominant_dampener == "persona_relevance"
    assert "answer_with_caveat" in breakdown.next_actions
    assert "check_other_perspective" in breakdown.next_actions


def test_compounding_weakness_asks_a_question():
    breakdown = build_breakdown(
        base_signal=0.8,
        hierarchy_decay=0.5,
        persona_relevance=0.5,
        path_weight=1.0,
        source_confidence=1.0,
    )
    assert breakdown.tier == ConfidenceTier.NEEDS_CLARIFICATION
    assert breakdown.next_actions[0] == "ask_clarifying_question"


def test_total_failure_refuses_and_offers_the_next_best_action():
    breakdown = build_breakdown(
        base_signal=0.8,
        hierarchy_decay=0.5,
        persona_relevance=0.5,
        path_weight=0.5,
        source_confidence=0.5,
        alternative_personas=["student"],
    )
    assert breakdown.tier == ConfidenceTier.UNAVAILABLE
    assert "state_unavailable" in breakdown.next_actions
    assert "switch_persona:student" in breakdown.next_actions
    assert "check_shared_archive" in breakdown.next_actions


def test_dampeners_are_floored_so_one_bad_signal_does_not_zero_the_result():
    floor = 0.45
    breakdown = build_breakdown(base_signal=1.0, source_confidence=0.0, floor=floor)
    assert breakdown.total == round(floor, 4)
    assert breakdown.tier == ConfidenceTier.UNCERTAIN


def test_scores_are_clamped_into_range():
    assert build_breakdown(base_signal=5.0).total == 1.0
    assert build_breakdown(base_signal=-1.0).tier == ConfidenceTier.UNAVAILABLE


# -- citations --------------------------------------------------------------


def _cited_node(**overrides):
    defaults = dict(
        id="n1",
        title="Token Refresh",
        type=NodeType.CONCEPT,
        content="Refresh your token before it expires.",
        sources=[
            SourceRef(
                uri="docs/auth.md",
                path="docs/auth.md",
                line_start=6,
                line_end=7,
                content_type="markdown",
                excerpt="Refresh your token before it expires.",
            )
        ],
    )
    defaults.update(overrides)
    return GraphNode(**defaults)


def test_citation_reports_path_and_line_range():
    citations = citations_for_node(_cited_node())
    assert len(citations) == 1
    assert citations[0].location() == "docs/auth.md:6-7"
    assert citations[0].content_type == "markdown"
    # unverifiable without source text, and it says so instead of pretending
    assert citations[0].verified is None


def test_verifier_confirms_a_correct_line_range():
    source = "line1\nline2\nline3\nline4\nline5\nRefresh your token before it expires.\nmore\n"
    verifier = SourceVerifier({"docs/auth.md": source})
    citations = citations_for_node(_cited_node(), verifier=verifier)
    assert citations[0].verified is True


def test_verifier_rejects_a_wrong_line_range():
    source = "line1\nline2\nsomething else entirely\n"
    verifier = SourceVerifier({"docs/auth.md": source})
    citations = citations_for_node(_cited_node(), verifier=verifier)
    assert citations[0].verified is False


def test_node_without_sources_still_cites_its_document():
    node = GraphNode(
        id="n2",
        title="Old Node",
        type=NodeType.CONCEPT,
        content="legacy content",
        source_uri="docs/legacy.md",
        locator="heading-lvl2:Legacy",
    )
    citations = citations_for_node(node)
    assert citations[0].location() == "docs/legacy.md"


# -- end to end -------------------------------------------------------------


def _taxonomy():
    return TagTaxonomy.from_dict(
        {
            "tags": [
                {"name": "authentication", "aliases": [{"alias": "login", "confidence": 0.95}]},
            ]
        }
    )


AUTH_DOC = """---
title: Authentication
tags: [login]
audience: developer
---
# Authentication

How sessions work.

## Token Refresh

Refresh your token before it expires.

```python
client.refresh()
```
"""


def test_alias_query_resolves_through_the_graph_and_keeps_provenance():
    client = RetellDB(use_memory=True, taxonomy=_taxonomy())
    client.ingest(
        content=AUTH_DOC,
        format="markdown",
        audience="developer",
        source_uri="docs/auth.md",
    )

    result = client.query("login", audience="developer", top_k=5)

    assert result.retrieval_path.value == "tag_traversal"
    titles = [sn.node.title for sn in result.nodes]
    assert "Token Refresh" in titles

    resolutions = [r for r in result.tag_resolutions if r.raw == "login"]
    assert resolutions and resolutions[0].canonical == "authentication"
    assert resolutions[0].rule.value == "alias"

    # the path that pulled the node in is auditable
    assert result.paths
    assert "chain_weight" in result.paths[0]

    # and the citation points at the real line range in the real file
    assert result.citations
    assert str(result.citations[0].location()).startswith("docs/auth.md:")
    assert result.confidence is not None
    assert result.confidence.tier in (
        ConfidenceTier.HIGH,
        ConfidenceTier.UNCERTAIN,
    )


def test_private_persona_memory_never_leaks_to_another_persona():
    client = RetellDB(use_memory=True, taxonomy=_taxonomy())
    client.ingest(
        content="""---
title: Student Deadlines
tags: [login]
audience: developer
---
# Student Deadlines

Essay due friday for the student persona.
""",
        format="markdown",
        audience="developer",
        source_uri="docs/student.md",
        personas=["student"],
        sensitivity="private",
    )

    result = client.query("login", audience="developer", persona="founder", top_k=5)

    assert result.nodes == []
    reasons = {b.reason for b in result.blocked}
    assert BlockReason.PRIVATE_OTHER_PERSONA in reasons
    assert result.confidence.tier == ConfidenceTier.UNAVAILABLE
    assert "switch_persona:student" in result.confidence.next_actions


def test_the_same_memory_is_visible_to_its_own_persona():
    client = RetellDB(use_memory=True, taxonomy=_taxonomy())
    client.ingest(
        content=AUTH_DOC,
        format="markdown",
        audience="developer",
        source_uri="docs/auth.md",
        personas=["student"],
        sensitivity="private",
    )

    result = client.query("login", audience="developer", persona="student", top_k=3)
    assert result.nodes
    assert any("Token Refresh" in sn.node.title for sn in result.nodes)
    assert result.blocked == []


def test_citations_are_verified_when_source_text_is_supplied():
    source = AUTH_DOC
    client = RetellDB(
        use_memory=True,
        taxonomy=_taxonomy(),
        source_verifier=SourceVerifier({"docs/auth.md": source}),
    )
    client.ingest(
        content=source, format="markdown", audience="developer", source_uri="docs/auth.md"
    )

    result = client.query("login", audience="developer", top_k=3)
    assert result.citations
    assert any(c.verified is True for c in result.citations)
