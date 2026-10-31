"""
Tests for latent association proposals.

The contract under test: embeddings propose during a build, a human accepts, and the graph
only ever holds reviewed weights. So these tests hold four lines: a below-threshold pair is
silence, a configured weight is never overwritten, an exclusion is never bridged in either
direction, and a missing neural embedder fails instead of quietly using token overlap.
"""

import math

import pytest

from retelldb.wrangling.latent import (
    LatentError,
    apply_proposals,
    cosine_similarity,
    embed_taxonomy,
    embedding_texts,
    parse_embedding_response,
    propose_associations,
    proposals_to_yaml,
    tag_surface,
)
from retelldb.wrangling.taxonomy import TagTaxonomy


def _taxonomy(*entries):
    return TagTaxonomy.from_dict({"tags": list(entries)})


def _vec(angle_degrees):
    radians = math.radians(angle_degrees)
    return [math.cos(radians), math.sin(radians)]


class FakeEmbedder:
    """Deterministic embedder keyed by surface text, for batcher tests."""

    name = "fake:test"

    def __init__(self, table):
        self.table = table
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self.table[text] for text in texts]


# ===========================================================================
# SIMILARITY
# ===========================================================================


class TestCosine:
    def test_identical_and_orthogonal(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
        assert cosine_similarity([1.0, 0.0], [0.0, 5.0]) == pytest.approx(0.0)

    def test_zero_vector_is_zero_not_nan(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_dimension_mismatch_is_an_error(self):
        with pytest.raises(LatentError, match="one embedder, one dimension"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# ===========================================================================
# PROPOSALS
# ===========================================================================


class TestProposals:
    def test_threshold_is_the_only_door(self):
        taxonomy = _taxonomy({"name": "auth"}, {"name": "billing"}, {"name": "tokens"})
        vectors = {
            "auth": _vec(0),
            "billing": _vec(80),    # cosine about 0.17, below the uncertain tier
            "tokens": _vec(30),     # cosine about 0.87, comfortably above
        }
        report = propose_associations(taxonomy, vectors, embedder="fake")

        pairs = {(p.source, p.target) for p in report.proposals}
        assert ("auth", "tokens") in pairs
        assert ("auth", "billing") not in pairs
        assert report.below_threshold == 1
        assert report.pairs_compared == 3
        # the rejected pair still moved the corpus statistics
        assert 0.0 < report.similarity_median <= report.similarity_p90

    def test_each_pair_is_proposed_once_with_the_raw_similarity_kept(self):
        taxonomy = _taxonomy({"name": "api"}, {"name": "sdk"})
        vectors = {"api": _vec(0), "sdk": _vec(10)}
        report = propose_associations(taxonomy, vectors, embedder="fake")

        assert len(report.proposals) == 1
        proposal = report.proposals[0]
        assert proposal.similarity == pytest.approx(0.9848, abs=1e-3)
        assert proposal.strength == round(proposal.similarity, 3)
        assert proposal.tier == "high"
        assert proposal.embedder == "fake"

    def test_a_configured_weight_is_never_overwritten(self):
        taxonomy = _taxonomy(
            {"name": "api", "related": {"sdk": 0.2}},
            {"name": "sdk"},
        )
        vectors = {"api": _vec(0), "sdk": _vec(5)}
        report = propose_associations(taxonomy, vectors, embedder="fake")

        assert report.proposals == []
        assert report.skipped_already_related == 1

    def test_exclusions_are_never_bridged_in_either_direction(self):
        taxonomy = _taxonomy(
            {"name": "fiction", "excludes": ["non fiction"]},
            {"name": "non fiction"},
        )
        vectors = {"fiction": _vec(0), "non fiction": _vec(2)}
        report = propose_associations(taxonomy, vectors, embedder="fake")

        assert report.proposals == []
        assert report.blocked_by_exclusion == 1

    def test_per_tag_budget_keeps_the_strongest_neighbours(self):
        taxonomy = _taxonomy(
            {"name": "hub"},
            {"name": "near"},
            {"name": "mid"},
            {"name": "far"},
        )
        vectors = {
            "hub": _vec(0),
            "near": _vec(10),   # 0.985
            "mid": _vec(45),    # 0.707
            "far": _vec(78),    # 0.208
        }
        report = propose_associations(
            taxonomy, vectors, threshold=0.5, max_per_tag=1, embedder="fake"
        )

        # the budget is per tag, so the far-away pair can still claim its own slot
        assert len(report.proposals) == 2
        assert [p.target for p in report.by_tag()["hub"]] == ["near"]
        assert "hub" not in {p.source for p in report.proposals if p.target in ("mid", "far")}

    def test_missing_vectors_are_missing_evidence(self):
        taxonomy = _taxonomy({"name": "auth"}, {"name": "tokens"})
        report = propose_associations(taxonomy, {"auth": _vec(0)}, embedder="fake")

        assert report.proposals == []
        assert report.tags_embedded == 1

    def test_no_embeddings_is_an_error(self):
        with pytest.raises(LatentError, match="from nothing"):
            propose_associations(_taxonomy({"name": "auth"}), {})

    def test_embeddings_for_unknown_tags_are_an_error(self):
        with pytest.raises(LatentError, match="no embedding matched"):
            propose_associations(_taxonomy({"name": "auth"}), {"billing": _vec(0)})

    def test_mixed_dimensions_are_an_error(self):
        taxonomy = _taxonomy({"name": "auth"}, {"name": "tokens"})
        with pytest.raises(LatentError, match="mixed embedding dimensions"):
            propose_associations(
                taxonomy, {"auth": [1.0, 0.0], "tokens": [1.0, 0.0, 0.0]}
            )

    def test_proposals_serialise_with_provenance(self):
        taxonomy = _taxonomy({"name": "api"}, {"name": "sdk"})
        report = propose_associations(
            taxonomy, {"api": _vec(0), "sdk": _vec(4)}, embedder="featherless:test"
        )
        yaml_text = proposals_to_yaml(report)

        assert "featherless:test" in yaml_text
        assert "dimension: 2" in yaml_text
        assert "not graph edges" in yaml_text


# ===========================================================================
# APPLYING
# ===========================================================================


class TestApplying:
    def test_edges_land_symmetrically(self):
        taxonomy = _taxonomy({"name": "api"}, {"name": "sdk"})
        report = propose_associations(
            taxonomy, {"api": _vec(0), "sdk": _vec(6)}, embedder="fake"
        )
        apply_proposals(taxonomy, report)

        by_name = {tag.name: tag for tag in taxonomy.tags}
        assert by_name["api"].related["sdk"] == report.proposals[0].strength
        assert by_name["sdk"].related["api"] == report.proposals[0].strength

    def test_an_earlier_weight_survives_a_second_pass(self):
        taxonomy = _taxonomy({"name": "api"}, {"name": "sdk"})
        vectors = {"api": _vec(0), "sdk": _vec(6)}
        apply_proposals(taxonomy, propose_associations(taxonomy, vectors, embedder="fake"))
        first = {tag.name: dict(tag.related) for tag in taxonomy.tags}

        apply_proposals(taxonomy, propose_associations(taxonomy, vectors, embedder="fake"))

        assert {tag.name: dict(tag.related) for tag in taxonomy.tags} == first

    def test_applying_refuses_to_bridge_an_exclusion(self):
        taxonomy = _taxonomy(
            {"name": "fiction", "excludes": ["non fiction"]},
            {"name": "non fiction"},
        )
        vectors = {"fiction": _vec(0), "non fiction": _vec(1)}
        report = propose_associations(taxonomy, vectors, embedder="fake")
        apply_proposals(taxonomy, report)

        assert all(tag.related == {} for tag in taxonomy.tags)

    def test_excludes_round_trip_through_yaml(self, tmp_path):
        from retelldb.wrangling.generator import taxonomy_to_yaml

        taxonomy = _taxonomy({"name": "fiction", "excludes": ["non fiction"]})
        path = tmp_path / "taxonomy.yaml"
        path.write_text(taxonomy_to_yaml(taxonomy), encoding="utf-8")

        loaded = TagTaxonomy.from_file(path)
        assert loaded.tags[0].excludes == ["non fiction"]


# ===========================================================================
# EMBEDDING LANE
# ===========================================================================


class TestEmbeddingLane:
    def test_surface_text_carries_the_aliases_humans_typed(self):
        taxonomy = _taxonomy(
            {"name": "authentication", "aliases": ["auth", "login"]},
            {"name": "billing"},
        )
        assert tag_surface(taxonomy.tags[0]) == "authentication: auth, login"
        assert embedding_texts(taxonomy) == ["authentication: auth, login", "billing"]

    def test_embedding_vectors_attach_to_the_tag_that_was_embedded(self):
        taxonomy = _taxonomy(
            {"name": "authentication", "aliases": ["auth"]},
            {"name": "billing"},
        )
        embedder = FakeEmbedder(
            {
                "authentication: auth": _vec(0),
                "billing": _vec(45),
            }
        )
        vectors = embed_taxonomy(embedder, taxonomy, batch_size=1)

        assert set(vectors) == {"authentication", "billing"}
        assert vectors["authentication"] == _vec(0)
        assert len(embedder.calls) == 2

    def test_a_short_embedding_response_is_an_error(self):
        from retelldb.wrangling.latent import LatentError as Error

        class ShortEmbedder:
            name = "short:test"

            def embed(self, texts):
                return [_vec(0)]

        with pytest.raises(Error, match="returned 1 vectors for 2 tags"):
            embed_taxonomy(ShortEmbedder(), _taxonomy({"name": "a"}, {"name": "b"}))

    def test_parse_embedding_response_is_index_ordered(self):
        payload = {"data": [{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]}
        parsed = parse_embedding_response(payload, ["first", "second"])

        assert parsed == {"first": [1.0, 2.0], "second": [3.0, 4.0]}

    def test_parse_embedding_response_rejects_a_mismatched_count(self):
        with pytest.raises(LatentError, match="items for 2 tags"):
            parse_embedding_response({"data": [{"embedding": [1.0]}]}, ["a", "b"])

    def test_featherless_refuses_without_a_key(self, monkeypatch):
        from retelldb.buildtime import featherless

        for name in featherless.API_KEY_ENV:
            monkeypatch.delenv(name, raising=False)
        embedder = featherless.FeatherlessEmbedder()

        assert embedder.available() is False
        with pytest.raises(featherless.EmbeddingUnavailable, match="no token-projection"):
            embedder.embed(["auth"])

    def test_featherless_refuses_in_a_production_deployment(self, monkeypatch):
        monkeypatch.setenv("RETELLDB_DEPLOYMENT", "production")
        from retelldb.buildtime.featherless import FeatherlessEmbedder

        with pytest.raises(RuntimeError, match="build-time only"):
            FeatherlessEmbedder(api_key="k")

    def test_featherless_sends_one_batch_and_checks_the_dimension(self, monkeypatch):
        import httpx
        from retelldb.buildtime import featherless

        seen = {}

        class _Response:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "data": [
                        {"embedding": [0.1] * featherless.MODEL_DIMENSIONS[featherless.DEFAULT_MODEL]}
                        for _ in range(2)
                    ]
                }

        def fake_post(url, json=None, headers=None, timeout=None):
            seen["url"] = url
            seen["json"] = json
            seen["headers"] = headers
            return _Response()

        monkeypatch.setattr(httpx, "post", fake_post)
        embedder = featherless.FeatherlessEmbedder(api_key="k")
        vectors = embedder.embed(["auth", "billing"])

        assert seen["url"] == f"{featherless.DEFAULT_BASE_URL}/embeddings"
        assert seen["json"] == {"model": featherless.DEFAULT_MODEL, "input": ["auth", "billing"]}
        assert seen["headers"]["User-Agent"].startswith("Mozilla/5.0")
        assert len(vectors) == 2
        assert len(vectors[0]) == featherless.MODEL_DIMENSIONS[featherless.DEFAULT_MODEL]


# ===========================================================================
# CLI WIRING
# ===========================================================================


class TestAssociateCommand:
    """The command embeds, proposes and stops. --apply is the only thing that writes edges."""

    @staticmethod
    def _fake_embedder(monkeypatch, table):
        from retelldb.buildtime import featherless

        class _Embedder:
            name = "featherless:fake"

            def __init__(self, model=None, **kwargs):
                self.model = model

            def embed(self, texts):
                return [table[text] for text in texts]

        monkeypatch.setattr(featherless, "FeatherlessEmbedder", _Embedder)

    def _write_taxonomy(self, tmp_path):
        path = tmp_path / "taxonomy.yaml"
        path.write_text(
            "tags:\n"
            "- name: api\n"
            "- name: sdk\n"
            "- name: billing\n"
            "- name: fiction\n"
            "  excludes:\n"
            "  - non fiction\n"
            "- name: non fiction\n",
            encoding="utf-8",
        )
        return path

    def test_proposals_are_written_and_no_edge_is_applied(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from retelldb.cli import main
        from retelldb.wrangling.taxonomy import TagTaxonomy

        self._fake_embedder(
            monkeypatch,
            {
                "api": _vec(0),
                "sdk": _vec(8),
                "billing": _vec(85),
                # the excluded pair is close only to itself, so it cannot ride in on a
                # transitive similarity to api
                "fiction": _vec(170),
                "non fiction": _vec(171),
            },
        )
        taxonomy_path = self._write_taxonomy(tmp_path)
        before = taxonomy_path.read_text(encoding="utf-8")
        out = tmp_path / "associations.yaml"

        result = CliRunner().invoke(
            main, ["taxonomy", "associate", str(taxonomy_path), "--out", str(out)]
        )

        assert result.exit_code == 0, result.output
        assert "1 proposals" in result.output
        assert "blocked by an exclusion" in result.output
        assert "median" in result.output
        # proposals on disk, taxonomy untouched
        assert "provenance:" in out.read_text(encoding="utf-8")
        assert taxonomy_path.read_text(encoding="utf-8") == before
        assert all(not tag.related for tag in TagTaxonomy.from_file(taxonomy_path).tags)

    def test_apply_writes_symmetric_edges_and_keeps_the_exclusion_out(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from retelldb.cli import main
        from retelldb.wrangling.taxonomy import TagTaxonomy

        self._fake_embedder(
            monkeypatch,
            {
                "api": _vec(0),
                "sdk": _vec(8),
                "billing": _vec(85),
                # the excluded pair is close only to itself, so it cannot ride in on a
                # transitive similarity to api
                "fiction": _vec(170),
                "non fiction": _vec(171),
            },
        )
        taxonomy_path = self._write_taxonomy(tmp_path)

        result = CliRunner().invoke(
            main,
            [
                "taxonomy", "associate", str(taxonomy_path),
                "--out", str(tmp_path / "associations.yaml"),
                "--apply",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Applied 1 associations" in result.output
        loaded = TagTaxonomy.from_file(taxonomy_path)
        by_name = {tag.name: tag for tag in loaded.tags}
        assert by_name["api"].related["sdk"] == pytest.approx(0.990, abs=1e-2)
        assert by_name["sdk"].related["api"] == pytest.approx(0.990, abs=1e-2)
        assert by_name["fiction"].related == {}
        assert by_name["non fiction"].related == {}

    def test_a_missing_embedder_fails_loudly_instead_of_proposing_nothing(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from retelldb.cli import main
        from retelldb.buildtime import featherless

        for name in featherless.API_KEY_ENV:
            monkeypatch.delenv(name, raising=False)
        taxonomy_path = self._write_taxonomy(tmp_path)

        result = CliRunner().invoke(
            main, ["taxonomy", "associate", str(taxonomy_path), "--out", str(tmp_path / "a.yaml")]
        )

        assert result.exit_code != 0
        assert "no token-projection" in result.output
        assert not (tmp_path / "a.yaml").exists()
