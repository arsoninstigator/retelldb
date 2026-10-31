"""
Tests for the canonical tag ontology and taxonomy generation.

The contract under test: an alias never replaces the raw tag, ambiguity resolves to
nothing, and a generator that fails never leaves a half-built ontology behind.
"""

import json

import pytest

from retelldb.wrangling.generator import (
    AdaptionTransport,
    GenerationError,
    build_prompt,
    collect_inventory,
    generate_taxonomy,
    parse_taxonomy,
    reconcile,
    taxonomy_to_yaml,
)
from retelldb.wrangling.taxonomy import ResolutionRule, TagTaxonomy, normalize_tag


def _taxonomy():
    return TagTaxonomy.from_dict(
        {
            "tags": [
                {"name": "programming", "aliases": ["dev", "coding"], "related": {"scripting": 0.6}},
                {"name": "scripting", "aliases": ["script"], "parents": ["programming"]},
                {"name": "authentication", "aliases": [{"alias": "auth", "confidence": 0.9}]},
                {"name": "billing", "aliases": ["auth"]},  # deliberate ambiguity
            ]
        }
    )


# -- resolution -------------------------------------------------------------


def test_exact_canonical_resolves_at_full_confidence():
    resolution = _taxonomy().resolve("Programming")
    assert resolution.rule == ResolutionRule.EXACT
    assert resolution.canonical == "programming"
    assert resolution.confidence == 1.0
    assert resolution.is_alias is False


def test_alias_resolves_without_losing_the_raw_tag():
    resolution = _taxonomy().resolve("dev")
    assert resolution.rule == ResolutionRule.ALIAS
    assert resolution.canonical == "programming"
    assert resolution.confidence == 0.95
    # the raw string is still available, unchanged
    assert resolution.raw == "dev"
    assert resolution.resolved is True


def test_ambiguous_alias_resolves_to_nothing():
    resolution = _taxonomy().resolve("auth")
    assert resolution.rule == ResolutionRule.AMBIGUOUS
    assert resolution.canonical is None
    assert resolution.resolved is False
    assert resolution.candidates == ["authentication", "billing"]


def test_unknown_tag_stays_its_own_canonical():
    resolution = _taxonomy().resolve("Chunky Salsa")
    assert resolution.rule == ResolutionRule.UNRESOLVED
    assert resolution.canonical == "chunky salsa"
    assert resolution.resolved is False


def test_parents_and_normalization():
    taxonomy = _taxonomy()
    assert taxonomy.parents_of("scripting") == ["programming"]
    assert normalize_tag("  Dev_Tools ") == "dev tools"


def test_empty_taxonomy_never_invents_canonicalisation():
    empty = TagTaxonomy()
    resolution = empty.resolve("dev")
    assert resolution.canonical == "dev"
    assert resolution.rule == ResolutionRule.UNRESOLVED


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "taxonomy.yaml"
    path.write_text(taxonomy_to_yaml(_taxonomy()), encoding="utf-8")
    reloaded = TagTaxonomy.from_file(path)
    assert set(reloaded.canonical_names()) == set(_taxonomy().canonical_names())
    assert reloaded.resolve("dev").canonical == "programming"


# -- generation -------------------------------------------------------------


class FakeTransport:
    """Stand-in for the provider: returns scripted text, records prompts."""

    def __init__(self, payload=None, available=True, error=None):
        self.payload = payload
        self._available = available
        self.error = error
        self.prompts = []

    def available(self):
        return self._available

    def complete(self, system, prompt):
        self.prompts.append(prompt)
        if self.error:
            return self.error
        return json.dumps(self.payload)


def test_generate_taxonomy_covers_every_observed_tag():
    inventory = collect_inventory(["dev", "coding", "billing", "weird-tag", "dev"])
    assert inventory.tags == ["billing", "coding", "dev", "weird tag"]

    payload = {
        "tags": [
            {"name": "programming", "aliases": [{"alias": "dev", "confidence": 0.97}, {"alias": "coding"}]},
            {"name": "billing", "aliases": []},
            {"name": "weird tag", "aliases": []},
        ],
        "ambiguous": [],
    }
    taxonomy, report = generate_taxonomy(inventory, FakeTransport(payload))

    assert taxonomy.resolve("dev").canonical == "programming"
    assert report["uncovered_raw_tags"] == []
    assert report["canonical_count"] == 3
    assert report["alias_count"] == 2


def test_generate_taxonomy_reports_uncovered_and_dangling():
    inventory = collect_inventory(["dev", "orphan"])
    payload = {
        "tags": [{"name": "programming", "aliases": ["dev"], "related": {"ghost": 0.5}}],
        "ambiguous": [{"raw": "auth", "candidates": ["authentication", "billing"]}],
    }
    _, report = generate_taxonomy(inventory, FakeTransport(payload))
    assert report["uncovered_raw_tags"] == ["orphan"]
    assert report["dangling_references"] == ["ghost"]
    assert report["ambiguous"][0]["raw"] == "auth"


def test_generation_fails_without_a_key():
    with pytest.raises(GenerationError):
        generate_taxonomy(collect_inventory(["dev"]), AdaptionTransport(api_key=""))


def test_generation_fails_when_the_provider_returns_nothing():
    with pytest.raises(GenerationError):
        generate_taxonomy(collect_inventory(["dev"]), FakeTransport(error=None))


def test_generation_fails_on_non_json():
    with pytest.raises(GenerationError):
        generate_taxonomy(collect_inventory(["dev"]), FakeTransport(error="sure! here you go"))


def test_parse_taxonomy_rejects_missing_tags_key():
    with pytest.raises(GenerationError):
        parse_taxonomy(json.dumps({"nope": True}))


def test_parse_taxonomy_clamps_alias_confidence_and_strips_fences():
    text = "```json\n" + json.dumps(
        {"tags": [{"name": "auth tag", "aliases": [{"alias": "a", "confidence": 4.2}]}]}
    ) + "\n```"
    taxonomy, ambiguous = parse_taxonomy(text)
    assert taxonomy.tags[0].aliases[0].confidence == 0.99
    assert ambiguous == []


def test_reconcile_flags_alias_colliding_with_a_canonical_name():
    inventory = collect_inventory(["auth", "authentication"])
    payload = {
        "tags": [
            {"name": "authentication", "aliases": []},
            {"name": "auth", "aliases": [{"alias": "authentication", "confidence": 0.8}]},
        ]
    }
    taxonomy, _ = parse_taxonomy(json.dumps(payload))
    report = reconcile(taxonomy, inventory)
    assert report["alias_collisions"] == ["authentication"]


def test_prompt_states_the_no_drop_rule():
    prompt = build_prompt(["dev", "billing"])
    assert "do not drop any" in prompt
    assert "ambiguous" in prompt
