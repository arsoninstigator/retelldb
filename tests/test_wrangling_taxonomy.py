"""
Tests for the canonical tag ontology and taxonomy generation.

The contract under test: an alias never replaces the raw tag, ambiguity resolves to
nothing, and a generator that fails never leaves a half-built ontology behind.
"""

import json

import pytest

from retelldb.buildtime.adaption import AdaptionTransport
from retelldb.wrangling.generator import (
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
    taxonomy, report = generate_taxonomy(inventory, FakeTransport(payload))
    assert report["uncovered_raw_tags"] == ["orphan"]
    # an association to a tag the batch never produced is dropped, not left dangling
    assert report["dangling_references"] == []
    assert report["dropped_relations"] == 1
    assert all(not tag.related for tag in taxonomy.tags)
    assert report["ambiguous"][0]["raw"] == "auth"


def test_generation_fails_without_a_key(monkeypatch):
    monkeypatch.delenv("ADAPTIONLABS_API_KEY", raising=False)
    monkeypatch.delenv("ADAPTION_LABS_API_KEY", raising=False)
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


# ===========================================================================
# BUILD-TIME ONLY CONTAINMENT
# ===========================================================================


class TestBuildTimeContainment:
    """
    The provider is a build dependency, never a runtime one. These tests hold the line
    in the two places it can slip: an import that drags the package in, and a production
    deployment that would quietly make a paid network call.
    """

    def test_transport_refuses_to_initialise_in_production(self, monkeypatch):
        from retelldb.buildtime.adaption import assert_build_time

        for name, value in (
            ("RETELLDB_DEPLOYMENT", "production"),
            ("RETELLDB_ENV", "prod"),
            ("ENVIRONMENT", "live"),
        ):
            monkeypatch.delenv("RETELLDB_DEPLOYMENT", raising=False)
            monkeypatch.delenv("RETELLDB_ENV", raising=False)
            monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
            monkeypatch.delenv("ENVIRONMENT", raising=False)
            monkeypatch.setenv(name, value)
            with pytest.raises(RuntimeError, match="build-time only"):
                assert_build_time()
            with pytest.raises(RuntimeError, match="build-time only"):
                AdaptionTransport(api_key="pt_test")

    def test_transport_completes_and_refuses_if_the_environment_changes_under_it(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        transport = AdaptionTransport(api_key="pt_test")
        assert transport.available() is True

        monkeypatch.setenv("RETELLDB_DEPLOYMENT", "production")
        with pytest.raises(RuntimeError, match="build-time only"):
            transport.complete("system", "prompt")

    def test_runtime_import_never_reaches_the_buildtime_package(self):
        import subprocess
        import sys

        probe = (
            "import sys, retelldb, retelldb.client, retelldb.wrangling\n"
            "leaked = [m for m in sys.modules if m.startswith('retelldb.buildtime')]\n"
            "print(leaked)\n"
            "raise SystemExit(1 if leaked else 0)"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        assert result.returncode == 0, f"runtime import pulled build-time code in: {result.stdout}{result.stderr}"


class TestAdaptionEndpointWiring:
    """
    The hosted endpoint is versioned under /api/v1, insists on a model name, and answers with
    a Responses-shaped payload whose reasoning items must never reach the json parser.
    """

    def test_base_url_accepts_every_documented_form(self):
        from retelldb.buildtime.adaption import DEFAULT_BASE_URL, normalize_base_url

        for form in (
            DEFAULT_BASE_URL,
            f"{DEFAULT_BASE_URL}/",
            f"{DEFAULT_BASE_URL}/v1",
            f"{DEFAULT_BASE_URL}/api/v1",
            f"{DEFAULT_BASE_URL}/api/v1/",
            None,
            "",
        ):
            assert normalize_base_url(form) == DEFAULT_BASE_URL

    def test_request_names_a_model_and_the_versioned_route(self, monkeypatch):
        import httpx
        from retelldb.buildtime import adaption

        captured = {}

        class _Response:
            status_code = 200

            @staticmethod
            def json():
                return {"output_text": '{"tags": []}'}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Response()

        monkeypatch.setattr(httpx, "post", fake_post)
        transport = adaption.AdaptionTransport(api_key="pt_test")
        assert transport.complete("system", "prompt") == '{"tags": []}'

        assert captured["url"] == f"{adaption.DEFAULT_BASE_URL}/api/v1/responses"
        assert captured["json"]["model"] == adaption.DEFAULT_MODEL
        assert [m["role"] for m in captured["json"]["input"]] == ["system", "user"]
        assert captured["headers"]["Authorization"] == "Bearer pt_test"

    def test_base_url_override_keeps_one_version_prefix(self, monkeypatch):
        import httpx
        from retelldb.buildtime import adaption

        seen = {}

        class _Response:
            status_code = 200

            @staticmethod
            def json():
                return {"output_text": "ok"}

        def fake_post(url, **kwargs):
            seen["url"] = url
            return _Response()

        monkeypatch.setattr(httpx, "post", fake_post)
        transport = adaption.AdaptionTransport(api_key="k", base_url=f"{adaption.DEFAULT_BASE_URL}/v1")
        transport.complete("s", "p")
        assert seen["url"] == f"{adaption.DEFAULT_BASE_URL}/api/v1/responses"

    def test_reasoning_items_never_reach_the_parser(self):
        from retelldb.wrangling.generator import extract_text

        response = {
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "let me think about tags"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": '{"tags": [{"canonical": "auth"}]}'}],
                },
            ],
            "output_text": '{"tags": [{"canonical": "auth"}]}',
        }
        assert extract_text(response) == '{"tags": [{"canonical": "auth"}]}'

        only_reasoning = {"output": [{"type": "reasoning", "content": [{"text": "hmm"}]}]}
        assert extract_text(only_reasoning) is None

    def test_default_batch_size_fits_the_gateway_window(self):
        from retelldb.buildtime import adaption
        from retelldb.wrangling.generator import DEFAULT_BATCH_SIZE

        # measured: 12 tags finish in about 22s, 16 tags took 57s, 20 tags hit the 60s gateway
        assert 6 <= DEFAULT_BATCH_SIZE <= 12
        assert adaption.DEFAULT_TIMEOUT <= 60
