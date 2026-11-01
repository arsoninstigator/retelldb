"""
Tests for the AO3 relation rules and the semantic/latent mapping.

Every rule here is copied from otwcode/otwarchive's validations (app/models/tag.rb,
app/models/meta_tagging.rb) rather than invented: synonyms are one hop to a canonical, both
sides of a relation share a type, links are acyclic, and no similarity score is allowed to
create identity.
"""

import pytest

from retelldb.wrangling.latent import propose_associations
from retelldb.wrangling.relations import (
    ObservedDocument,
    unwrangled_buckets,
    SYNONYM_CANDIDATE_SIMILARITY,
    RelationViolationKind,
    check_relations,
    explain_mapping,
    synonym_candidates,
    types_compatible,
)
from retelldb.wrangling.taxonomy import TagTaxonomy


def _taxonomy(*entries):
    return TagTaxonomy.from_dict({"tags": list(entries)})


def _kinds(violations):
    return [violation.kind for violation in violations]


class FakeEmbedder:
    name = "fake:test"

    def __init__(self, table):
        self.table = table

    def embed(self, texts):
        return [self.table[text] for text in texts]


# ===========================================================================
# SYNONYM RULES (Tag#merger_id)
# ===========================================================================


class TestSynonymRules:
    def test_an_alias_that_is_also_canonical_is_a_violation(self):
        taxonomy = _taxonomy({"name": "auth", "aliases": ["login"]}, {"name": "login"})
        violations = check_relations(taxonomy)

        assert RelationViolationKind.ALIAS_SHADOWS_CANONICAL in _kinds(violations)
        assert "would shadow" in violations[0].detail

    def test_a_two_hop_synonym_is_caught_as_a_shadowed_canonical(self):
        # "coding" is both an alias of programming and its own canonical tag, which is the
        # only way a synonym-of-a-synonym can even be written in this format
        taxonomy = _taxonomy(
            {"name": "programming", "aliases": ["coding"]},
            {"name": "coding", "aliases": ["dev"]},
        )
        kinds = _kinds(check_relations(taxonomy))

        assert kinds == [RelationViolationKind.ALIAS_SHADOWS_CANONICAL]

    def test_a_tag_cannot_be_its_own_synonym(self):
        taxonomy = _taxonomy({"name": "auth", "aliases": ["Auth"]})
        assert _kinds(check_relations(taxonomy)) == [
            RelationViolationKind.SELF_ALIAS,
        ]

    def test_a_clean_one_hop_taxonomy_has_no_violations(self):
        taxonomy = _taxonomy(
            {"name": "programming", "aliases": ["coding", "dev"]},
            {"name": "testing"},
        )
        assert check_relations(taxonomy) == []


# ===========================================================================
# TYPE BOUNDARIES
# ===========================================================================


class TestTypeBoundaries:
    def test_two_known_types_must_match(self):
        people = _taxonomy({"name": "jane austen", "tag_type": "person"})
        works = _taxonomy({"name": "pride and prejudice", "tag_type": "work"})
        assert types_compatible(people.tags[0], works.tags[0]) is False

    def test_an_untyped_tag_never_blocks_a_relation(self):
        untyped = _taxonomy({"name": "auth"})
        typed = _taxonomy({"name": "authentication", "tag_type": "concept"})
        assert types_compatible(untyped.tags[0], typed.tags[0]) is True

    def test_cross_type_associations_are_violations(self):
        taxonomy = _taxonomy(
            {"name": "austen", "tag_type": "person", "related": {"emma": 0.7}},
            {"name": "emma", "tag_type": "work", "related": {"austen": 0.7}},
        )
        assert RelationViolationKind.CROSS_TYPE_ASSOCIATION in _kinds(check_relations(taxonomy))

    def test_latent_never_proposes_across_types(self):
        taxonomy = _taxonomy(
            {"name": "austen", "tag_type": "person"},
            {"name": "emma", "tag_type": "work"},
            {"name": "auth", "tag_type": "concept"},
            {"name": "authentication", "tag_type": "concept"},
        )
        vectors = {
            "austen": [1.0, 0.0],
            "emma": [0.999, 0.001],
            "auth": [0.0, 1.0],
            "authentication": [0.001, 0.999],
        }
        report = propose_associations(taxonomy, vectors, embedder="fake")

        pairs = {(p.source, p.target) for p in report.proposals}
        assert pairs == {("auth", "authentication")}
        # five of the six pairs in this fixture cross a type boundary
        assert report.skipped_cross_type == 5


# ===========================================================================
# ASSOCIATIONS (the only thing latent may create)
# ===========================================================================


class TestAssociationRules:
    def test_an_association_must_point_at_a_canonical_tag(self):
        taxonomy = _taxonomy({"name": "auth", "related": {"nope": 0.6}})
        assert RelationViolationKind.RELATED_TARGET_MISSING in _kinds(check_relations(taxonomy))

    def test_an_association_must_be_symmetric(self):
        taxonomy = _taxonomy(
            {"name": "auth", "related": {"tokens": 0.6}},
            {"name": "tokens", "related": {"auth": 0.4}},
        )
        assert RelationViolationKind.ASSOCIATION_NOT_SYMMETRIC in _kinds(check_relations(taxonomy))

    def test_a_tag_cannot_associate_with_itself(self):
        taxonomy = _taxonomy({"name": "auth", "related": {"auth": 0.9}})
        assert RelationViolationKind.SELF_ASSOCIATION in _kinds(check_relations(taxonomy))

    def test_an_excluded_pair_cannot_also_be_associated(self):
        taxonomy = _taxonomy(
            {"name": "fiction", "excludes": ["non fiction"], "related": {"non fiction": 0.5}},
            {"name": "non fiction", "related": {"fiction": 0.5}},
        )
        assert RelationViolationKind.EXCLUDED_AND_RELATED in _kinds(check_relations(taxonomy))


# ===========================================================================
# SEMANTIC VS LATENT
# ===========================================================================


class TestSemanticVsLatent:
    def _close_pair_taxonomy(self):
        return _taxonomy(
            {"name": "authentication", "tag_type": "concept"},
            {"name": "auth", "tag_type": "concept"},
        )

    def test_high_latent_proximity_is_only_a_candidate(self):
        taxonomy = self._close_pair_taxonomy()
        vectors = {"authentication": [1.0, 0.0], "auth": [0.9995, 0.03]}
        report = propose_associations(taxonomy, vectors, embedder="fake")

        candidates = synonym_candidates(taxonomy, report.proposals)

        assert len(candidates) == 1
        assert candidates[0].similarity >= SYNONYM_CANDIDATE_SIMILARITY
        # the taxonomy itself is untouched: a candidate is not an alias
        assert all(not tag.aliases for tag in taxonomy.tags)

    def test_a_configured_alias_never_comes_back_as_a_candidate(self):
        taxonomy = _taxonomy(
            {"name": "authentication", "tag_type": "concept", "aliases": ["auth"]},
            {"name": "auth", "tag_type": "concept"},
        )
        taxonomy.tags[1].aliases = []
        vectors = {"authentication": [1.0, 0.0], "auth": [0.9995, 0.03]}
        report = propose_associations(taxonomy, vectors, embedder="fake")

        # a tag whose name is already an alias of the other is not a candidate
        assert synonym_candidates(taxonomy, report.proposals) == []

    def test_mid_range_proximity_is_an_association_without_a_candidate(self):
        taxonomy = _taxonomy(
            {"name": "api", "tag_type": "concept"},
            {"name": "sdk", "tag_type": "concept"},
        )
        vectors = {"api": [1.0, 0.0], "sdk": [0.85, 0.52]}
        report = propose_associations(taxonomy, vectors, embedder="fake")

        assert len(report.proposals) == 1
        assert synonym_candidates(taxonomy, report.proposals) == []

    def test_cross_type_proximity_is_neither_an_association_nor_a_candidate(self):
        taxonomy = _taxonomy(
            {"name": "austen", "tag_type": "person"},
            {"name": "emma", "tag_type": "work"},
        )
        vectors = {"austen": [1.0, 0.0], "emma": [0.9999, 0.01]}
        report = propose_associations(taxonomy, vectors, embedder="fake")

        assert report.proposals == []
        assert synonym_candidates(taxonomy, report.proposals) == []

    def test_the_mapping_is_stated_in_one_place(self):
        lines = explain_mapping()

        assert any("semantic" in line and "synonym" in line for line in lines)
        assert any("never written as an alias" in line for line in lines)
        assert any("cross-type" in line for line in lines)


# ===========================================================================
# REPAIR
# ===========================================================================


class TestRepair:
    def test_a_one_sided_edge_is_mirrored(self):
        from retelldb.wrangling.relations import repair_relations

        taxonomy = _taxonomy({"name": "auth", "related": {"tokens": 0.7}}, {"name": "tokens"})
        counts = repair_relations(taxonomy)

        by_name = {tag.name: tag for tag in taxonomy.tags}
        assert by_name["tokens"].related["auth"] == 0.7
        assert counts["symmetric_added"] == 1
        assert check_relations(taxonomy) == []

    def test_a_disagreement_keeps_the_stronger_claim(self):
        from retelldb.wrangling.relations import repair_relations

        taxonomy = _taxonomy(
            {"name": "auth", "related": {"tokens": 0.4}},
            {"name": "tokens", "related": {"auth": 0.8}},
        )
        counts = repair_relations(taxonomy)

        by_name = {tag.name: tag for tag in taxonomy.tags}
        assert by_name["auth"].related["tokens"] == 0.8
        assert by_name["tokens"].related["auth"] == 0.8
        assert counts["symmetric_added"] == 1

    def test_an_edge_to_nowhere_is_dropped_and_counted(self):
        from retelldb.wrangling.relations import repair_relations

        taxonomy = _taxonomy({"name": "auth", "related": {"ghost": 0.6, "auth": 0.5}})
        counts = repair_relations(taxonomy)

        assert taxonomy.tags[0].related == {}
        assert counts["dropped_relations"] == 2

    def test_a_clean_taxonomy_is_left_alone(self):
        from retelldb.wrangling.relations import repair_relations

        taxonomy = _taxonomy(
            {"name": "auth", "related": {"tokens": 0.7}},
            {"name": "tokens", "related": {"auth": 0.7}},
        )
        assert repair_relations(taxonomy) == {
            "symmetric_added": 0,
            "dropped_relations": 0,
            "dropped_self_aliases": 0,
        }

    def test_a_tag_aliased_to_itself_is_dropped(self):
        from retelldb.wrangling.relations import repair_relations

        taxonomy = _taxonomy({"name": "http", "aliases": ["HTTP", "http requests"]})
        counts = repair_relations(taxonomy)

        assert [rule.alias for rule in taxonomy.tags[0].aliases] == ["http requests"]
        assert counts["dropped_self_aliases"] == 1
        assert check_relations(taxonomy) == []


class TestUnwrangledBuckets:
    """
    otwarchive: synning a tag to a canonical moves its unwrangled neighbours into the
    canonical's bins, de-synning takes them out. The grouping signal is co-occurrence.
    """

    def test_tags_are_grouped_under_the_canonicals_they_were_seen_with(self):
        report = unwrangled_buckets(
            [
                ObservedDocument(
                    canonical_tags=["authentication"], unwrangled_tags=["auth flow", "sso"]
                ),
                ObservedDocument(canonical_tags=["billing"], unwrangled_tags=["invoice"]),
            ]
        )

        assert report.buckets == {
            "authentication": ["auth flow", "sso"],
            "billing": ["invoice"],
        }
        assert report.unattributed == []

    def test_a_tag_with_nothing_nearby_is_unattributed_not_guessed(self):
        report = unwrangled_buckets(
            [ObservedDocument(canonical_tags=[], unwrangled_tags=["stray thought"])]
        )

        assert report.buckets == {}
        assert report.unattributed == ["stray thought"]

    def test_the_same_tag_lands_in_every_bucket_it_rides_with(self):
        report = unwrangled_buckets(
            [
                ObservedDocument(
                    canonical_tags=["api", "sdk"], unwrangled_tags=["rate limit"]
                )
            ]
        )

        assert report.buckets == {"api": ["rate limit"], "sdk": ["rate limit"]}

    def test_documents_without_unwrangled_tags_add_no_buckets(self):
        report = unwrangled_buckets([ObservedDocument(canonical_tags=["api"])])

        assert report.buckets == {}
        assert report.unattributed == []
        assert report.documents == 1


class TestBinsCommand:
    """The bins command is the reviewed-ontology workflow: new docs against an old ontology."""

    def test_tags_the_ontology_does_not_cover_are_grouped_by_their_neighbours(self, tmp_path):
        from click.testing import CliRunner

        from retelldb.cli import main

        taxonomy_file = tmp_path / "taxonomy.yaml"
        taxonomy_file.write_text(
            "tags:\n"
            "  - name: authentication\n"
            "    aliases:\n"
            "      - alias: login\n"
            "  - name: billing\n",
            encoding="utf-8",
        )
        corpus = tmp_path / "docs"
        corpus.mkdir()
        (corpus / "auth.md").write_text(
            "# Authentication\n\nLogin tokens and session handling.\n", encoding="utf-8"
        )
        (corpus / "billing.md").write_text(
            "# Billing\n\nInvoice reminders and payment plans.\n", encoding="utf-8"
        )

        result = CliRunner().invoke(main, ["taxonomy", "bins", str(taxonomy_file), str(corpus)])

        assert result.exit_code == 0, result.output
        assert "bucket(s)" in result.output
        assert "authentication <-" in result.output
        # the billing document carried a tag the ontology already covers, so it adds nothing
        assert "billing <-" not in result.output


class TestPersonaMemory:
    """The persona layer prixie and clawtel read, now part of the standalone package."""

    def test_a_voice_profile_round_trips_through_its_own_memory_graph(self):
        from retelldb import RetellDB
        from retelldb.models.persona import PersonaProfile
        from retelldb.persona_store import load_profile, save_profile

        client = RetellDB(use_memory=True)
        profile = PersonaProfile(persona_id="hackathon", display_name="Hackathon You")

        saved = save_profile(client.graph_store, client.config, profile)
        loaded = load_profile(client.graph_store, client.config, "hackathon")

        assert saved.success is True
        assert saved.graph == "memory_hackathon"
        assert loaded is not None and loaded.display_name == "Hackathon You"
        # another persona's graph knows nothing about this one
        assert load_profile(client.graph_store, client.config, "founder") is None

    def test_the_package_version_tracks_pyproject(self):
        import pathlib
        import re

        import retelldb

        pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        declared = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.M).group(1)
        assert retelldb.__version__ == declared
