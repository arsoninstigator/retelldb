"""
The two similarities, kept apart.

Archive of Our Own (otwcode/otwarchive, models app/models/tag.rb, meta_tagging.rb) is the
reference implementation of tag wrangling, and it is worth reading because of what it does
not have: any notion of similarity at all. No embeddings, no fuzzy matching, no automatic
merging. Its whole relation model is three explicit, human-authored links:

- **synonym** (the `merger_id` self-reference): a noncanonical tag whose `merger` is a
  canonical tag, meaning "same concept, different wording". `Tag#filter` returns the merger
  only when that merger is itself canonical, so a synonym never filters on its own.
- **meta** (`MetaTagging`, `meta_tag` -> `sub_tag`): "the meta tag is a superset of the sub
  tag". Not identity. Both sides must be canonical, both must be the same tag class, and a
  meta tag may not be its own grandparent.
- **unwrangleable**: a flag saying this tag should not be wrangled at all, which cannot be
  set on a canonical or a synonymized tag.

The lessons that map onto retelldb, and that this module encodes:

1. similarity is evidence about *wording*, so it can only ever justify a synonym proposal,
   never the link itself. AO3 merges by hand for a reason.
2. relations are type-bound. A synonym and a meta link both require the two tags to be the
   same category (AO3 class; here `tag_type`), which is what stops "jane austen" (a person)
   from merging into "pride and prejudice" (a work) just because they are close in space.
3. links are one hop and acyclic. A synonym's target is canonical, never another synonym, and
   a meta tag cannot be its own ancestor.
4. proximity without identity is an association, not a merge. Two tags can sit near each other
   in latent space and still be different concepts, which is exactly what an association with
   a strength is for.

So: semantic similarity (same words, alias rules, a human or model naming one canonical) may
create a synonym. Latent similarity may create an association and, at the top of its range, a
*flagged candidate* for a human to turn into a synonym. It may never create the synonym.
"""

from typing import Sequence, Dict, List, Optional

from pydantic import BaseModel, Field

from retelldb.wrangling.taxonomy import CanonicalTag, TagTaxonomy, normalize_tag

#: cosine above which latent proximity is worth a human's attention as a possible synonym
SYNONYM_CANDIDATE_SIMILARITY = 0.90


class RelationViolationKind(str):
    """Why a configured relation breaks the AO3 rules."""

    ALIAS_TARGET_MISSING = "alias_target_missing"
    ALIAS_SHADOWS_CANONICAL = "alias_shadows_canonical"
    SELF_ALIAS = "self_alias"
    CROSS_TYPE_SYNONYM = "cross_type_synonym"
    RELATED_TARGET_MISSING = "related_target_missing"
    SELF_ASSOCIATION = "self_association"
    ASSOCIATION_NOT_SYMMETRIC = "association_not_symmetric"
    CROSS_TYPE_ASSOCIATION = "cross_type_association"
    PARENT_TARGET_MISSING = "parent_target_missing"
    CROSS_TYPE_PARENT = "cross_type_parent"
    SELF_PARENT = "self_parent"
    PARENT_CYCLE = "parent_cycle"
    EXCLUSION_TARGET_MISSING = "exclusion_target_missing"
    EXCLUDED_AND_RELATED = "excluded_and_related"


class RelationViolation(BaseModel):
    """One broken relation, named well enough to fix without reading this file."""

    kind: str
    subject: str
    detail: str


class SynonymCandidate(BaseModel):
    """
    Latent proximity flagged for a human to judge.

    Deliberately not an edge and not an alias. `similarity` is the evidence, and the only way
    this becomes a synonym is a person writing the alias rule.
    """

    source: str
    target: str
    similarity: float
    tag_type: Optional[str] = None


def _alias_index(taxonomy: TagTaxonomy) -> Dict[str, List[str]]:
    """alias -> the canonical tags claiming it."""
    index: Dict[str, List[str]] = {}
    for tag in taxonomy.tags:
        for rule in tag.aliases:
            index.setdefault(normalize_tag(rule.alias), []).append(tag.normalized_name())
    return index


def check_relations(taxonomy: TagTaxonomy) -> List[RelationViolation]:
    """
    Check a taxonomy against the AO3 relation rules.

    Everything here is a rule otwarchive enforces in validations, expressed against retelldb's
    config format: same-type synonyms, canonical targets only, one hop, symmetric associations,
    and exclusions that do not co-exist with an association.
    """
    by_name = {tag.normalized_name(): tag for tag in taxonomy.tags}
    aliases = _alias_index(taxonomy)
    violations: List[RelationViolation] = []

    for tag in taxonomy.tags:
        name = tag.normalized_name()

        for rule in tag.aliases:
            alias = normalize_tag(rule.alias)
            if alias == name:
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.SELF_ALIAS,
                        subject=tag.name,
                        detail=f"'{rule.alias}' is this tag's own name",
                    )
                )
                continue
            if alias in by_name:
                # AO3: a synonym's merger must be canonical, never another synonym. in this
                # config format aliases only hang off canonical tags, so a two-hop chain shows
                # up exactly here: the alias name is itself a canonical tag.
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.ALIAS_SHADOWS_CANONICAL,
                        subject=tag.name,
                        detail=(
                            f"alias '{rule.alias}' is also a canonical tag, so it would "
                            "shadow that tag"
                        ),
                    )
                )

        for related, strength in tag.related.items():
            target = normalize_tag(related)
            if target == name:
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.SELF_ASSOCIATION,
                        subject=tag.name,
                        detail="a tag cannot be associated with itself",
                    )
                )
                continue
            other = by_name.get(target)
            if other is None:
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.RELATED_TARGET_MISSING,
                        subject=tag.name,
                        detail=f"association points at '{related}', which is not canonical",
                    )
                )
                continue
            if other.related.get(normalize_tag(tag.name)) != strength:
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.ASSOCIATION_NOT_SYMMETRIC,
                        subject=tag.name,
                        detail=(
                            f"association to '{other.name}' is {strength} here but "
                            f"{other.related.get(normalize_tag(tag.name))} there"
                        ),
                    )
                )
            if not types_compatible(tag, other):
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.CROSS_TYPE_ASSOCIATION,
                        subject=tag.name,
                        detail=(
                            f"association crosses tag types "
                            f"({tag.tag_type or 'untyped'} -> {other.tag_type or 'untyped'})"
                        ),
                    )
                )
            if target in {normalize_tag(x) for x in tag.excludes}:
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.EXCLUDED_AND_RELATED,
                        subject=tag.name,
                        detail=f"'{other.name}' is both excluded and associated",
                    )
                )

        for excluded in tag.excludes:
            if normalize_tag(excluded) not in by_name:
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.EXCLUSION_TARGET_MISSING,
                        subject=tag.name,
                        detail=f"exclusion names '{excluded}', which is not canonical",
                    )
                )

    # hierarchy rules, straight from MetaTagging's validations: both ends canonical, same
    # class, never itself, and "a metatag can't be its own grandparent" (no cycles).
    for tag in taxonomy.tags:
        name = tag.normalized_name()
        for parent in tag.parents:
            key = normalize_tag(parent)
            target = by_name.get(key)
            if target is None:
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.PARENT_TARGET_MISSING,
                        subject=name,
                        detail=f"parent '{key}' is not a canonical tag, so it is not a meta tag",
                    )
                )
            elif key == name:
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.SELF_PARENT,
                        subject=name,
                        detail="a tag cannot be its own meta tag",
                    )
                )
            elif not types_compatible(tag, target):
                violations.append(
                    RelationViolation(
                        kind=RelationViolationKind.CROSS_TYPE_PARENT,
                        subject=name,
                        detail=(
                            f"meta tag '{key}' is a {target.tag_type} while the sub tag is a "
                            f"{tag.tag_type}; meta tags must share the sub tag's type"
                        ),
                    )
                )

    for tag in taxonomy.tags:
        if _reaches_itself(tag.normalized_name(), by_name):
            violations.append(
                RelationViolation(
                    kind=RelationViolationKind.PARENT_CYCLE,
                    subject=tag.normalized_name(),
                    detail="following parents from this tag leads back to it (a metatag cannot be its own grandparent)",
                )
            )

    return violations


def _reaches_itself(start: str, by_name: Dict[str, CanonicalTag]) -> bool:
    """Walk the parent chain; True when it loops back to where it started."""
    frontier = [normalize_tag(p) for p in (by_name[start].parents if start in by_name else [])]
    seen: List[str] = []
    while frontier:
        current = frontier.pop()
        if current == start:
            return True
        if current in seen:
            continue
        seen.append(current)
        node = by_name.get(current)
        if node is not None:
            frontier.extend(normalize_tag(p) for p in node.parents)
    return False


def repair_relations(taxonomy: TagTaxonomy) -> Dict[str, int]:
    """
    Make the configured relations obey the rules instead of only reporting that they broke them.

    A generated taxonomy routinely writes an edge on one side only, or names a tag the batch
    never produced. AO3's associations are symmetric and canonical-only, so this drops the
    references that point nowhere and mirrors the rest. When both sides disagree about the
    strength, the stronger claim wins: it is the one a reviewer should look at.

    Returns counts, because a repair that happens silently is a repair nobody audits.
    """
    by_name = {tag.normalized_name(): tag for tag in taxonomy.tags}
    dropped = 0
    added = 0
    self_aliases = 0

    # a tag can't be its own synonym, and generators happily emit "http" aliased to "http"
    for tag in taxonomy.tags:
        kept_aliases = []
        for rule in tag.aliases:
            if normalize_tag(rule.alias) == tag.normalized_name():
                self_aliases += 1
                continue
            kept_aliases.append(rule)
        tag.aliases = kept_aliases

    for tag in taxonomy.tags:
        kept: Dict[str, float] = {}
        for target, strength in tag.related.items():
            normalized = normalize_tag(target)
            if normalized == tag.normalized_name() or normalized not in by_name:
                dropped += 1
                continue
            kept[target] = strength
        tag.related = kept

    for tag in taxonomy.tags:
        for target, strength in list(tag.related.items()):
            other = by_name[normalize_tag(target)]
            key = tag.normalized_name()
            current = other.related.get(key)
            if current is None:
                other.related[key] = strength
                added += 1
            elif abs(current - strength) > 1e-9:
                agreed = max(current, strength)
                other.related[key] = agreed
                tag.related[target] = agreed
                added += 1

    return {
        "symmetric_added": added,
        "dropped_relations": dropped,
        "dropped_self_aliases": self_aliases,
    }


def types_compatible(a: CanonicalTag, b: CanonicalTag) -> bool:
    """
    True when two tags may be related at all.

    AO3 requires both sides of a synonym and both sides of a meta link to be the same class.
    An untyped tag is not evidence of a mismatch, so it never blocks; but two known, different
    types always do.
    """
    if not a.tag_type or not b.tag_type:
        return True
    return a.tag_type.lower() == b.tag_type.lower()


def synonym_candidates(taxonomy: TagTaxonomy, proposals) -> List[SynonymCandidate]:
    """
    Pull the top of the latent range out as possible synonyms, never as synonyms.

    A high cosine between two differently named tags in the same type is exactly the evidence
    a wrangler would look at: close enough that it might be the same concept, far enough from
    proof that a human should decide. Anything that is already an alias, already canonical
    under another spelling, or explicitly excluded is filtered out first.
    """
    by_name = {tag.normalized_name(): tag for tag in taxonomy.tags}
    aliases = _alias_index(taxonomy)

    out: List[SynonymCandidate] = []
    for proposal in proposals:
        if proposal.similarity < SYNONYM_CANDIDATE_SIMILARITY:
            continue
        source = by_name.get(normalize_tag(proposal.source))
        target = by_name.get(normalize_tag(proposal.target))
        if source is None or target is None:
            continue
        if not types_compatible(source, target):
            continue
        if normalize_tag(target.name) in {normalize_tag(x) for x in source.excludes}:
            continue
        if normalize_tag(target.name) in _alias_names(source) or normalize_tag(
            source.name
        ) in _alias_names(target):
            # already spelled out as the same concept: nothing to review
            continue
        if normalize_tag(source.name) in aliases.get(normalize_tag(target.name), []):
            continue
        out.append(
            SynonymCandidate(
                source=source.name,
                target=target.name,
                similarity=round(proposal.similarity, 4),
                tag_type=source.tag_type,
            )
        )
    return out


def _alias_names(tag: CanonicalTag) -> set:
    return {normalize_tag(rule.alias) for rule in tag.aliases}


def explain_mapping() -> List[str]:
    """
    The mapping, in one place, so docs and code cannot disagree.

    Returned as lines rather than prose so the CLI can print exactly what the tests assert.
    """
    return [
        "semantic (same wording, alias rule, human or model naming the canonical): "
        "creates a synonym, which is the only relation that means 'same concept'",
        f"latent >= {SYNONYM_CANDIDATE_SIMILARITY}: a synonym candidate for a human to review, "
        "never written as an alias",
        "latent >= threshold: an association with its cosine as strength, symmetric",
        "latent below threshold: silence",
        "cross-type pairs: no relation at any similarity (AO3 types are load bearing)",
    ]


class ObservedDocument(BaseModel):
    """One document's tags, split by whether the wrangler could place them."""

    canonical_tags: List[str] = Field(default_factory=list)
    unwrangled_tags: List[str] = Field(default_factory=list)


class UnwrangledBuckets(BaseModel):
    """Unwrangled tags grouped by the canonical tag they were seen with."""

    buckets: Dict[str, List[str]] = Field(default_factory=dict)
    unattributed: List[str] = Field(default_factory=list)
    documents: int = 0


def unwrangled_buckets(documents: Sequence[ObservedDocument]) -> UnwrangledBuckets:
    """
    Attribute unwrangled tags to the canonicals they were observed with.

    otwarchive's spec says it plainly: "Synning a fandom to a canonical fandom moves its
    unwrangled tags to the canonical's unwrangled bins; de-synning takes them out"
    (features/tags_and_wrangling/tag_wrangling_more.feature). The mechanism is
    co-occurrence, not string similarity: an unwrangled tag rides along with the canonical
    tags on the same document, so that is how it gets a home to be reviewed in.

    A tag observed with no canonical at all stays unattributed, which is the honest answer:
    there is nothing to review it against yet.
    """
    buckets: Dict[str, Set[str]] = {}
    unattributed: Set[str] = set()

    for document in documents:
        anchors = sorted({normalize_tag(tag) for tag in document.canonical_tags if tag})
        loose = sorted({str(tag) for tag in document.unwrangled_tags if str(tag).strip()})
        if not loose:
            continue
        if not anchors:
            unattributed.update(loose)
            continue
        for anchor in anchors:
            buckets.setdefault(anchor, set()).update(loose)

    return UnwrangledBuckets(
        buckets={name: sorted(tags) for name, tags in sorted(buckets.items())},
        unattributed=sorted(unattributed),
        documents=len(documents),
    )
