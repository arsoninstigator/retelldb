"""
Latent similarity, kept in its place.

memorium demotes embeddings from "the thing that understands the knowledge base" to one
component of retrieval: associations are explicit weighted edges a human can inspect, and
vector distance is not allowed to invent them at query time. This module honours that by
doing the one job latent space is actually good at, and only during a build:

    embeddings propose association edges, a human accepts them into taxonomy.yaml.

So nothing here mutates a taxonomy. Embeddings arrive from a build-time lane (see
`retelldb.buildtime.featherless`), similarity becomes a proposal with its score kept as
provenance, and the strength that ends up in the graph is the reviewed number. A tag that
scores below the uncertain tier produces silence, not a weak edge dressed up as knowledge.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from retelldb.wrangling.confidence import tier_for
from retelldb.wrangling.taxonomy import CanonicalTag, TagTaxonomy, normalize_tag

#: below the "uncertain" tier a similarity is not a proposal, it is noise
MIN_SIMILARITY = 0.45

#: a tag with more than this many neighbours stops being a concept and starts being a hub
DEFAULT_MAX_PER_TAG = 4


class LatentError(RuntimeError):
    """Raised when latent proposals cannot be produced honestly."""


class TagEmbedding(BaseModel):
    """One canonical tag and its vector."""

    tag: str
    vector: List[float]


class AssociationProposal(BaseModel):
    """
    A proposed association edge, with the evidence that produced it.

    `similarity` is the raw cosine, kept so a reviewer can see what the embedder said.
    `strength` is what belongs in the taxonomy once accepted. `tier` mirrors the
    confidence tiers so a 0.5 proposal is visibly weaker than a 0.8 one.
    """

    source: str
    target: str
    similarity: float
    strength: float
    tier: str
    embedder: str


class LatentReport(BaseModel):
    """Everything a latent pass did, including what it refused to do."""

    embedder: str
    dimension: int
    threshold: float
    tags_embedded: int
    proposals: List[AssociationProposal] = Field(default_factory=list)
    skipped_already_related: int = 0
    blocked_by_exclusion: int = 0
    below_threshold: int = 0
    # where this corpus actually sits, so a threshold is chosen with evidence
    pairs_compared: int = 0
    similarity_median: float = 0.0
    similarity_p90: float = 0.0

    def by_tag(self) -> Dict[str, List[AssociationProposal]]:
        """Group proposals by source tag, strongest first."""
        grouped: Dict[str, List[AssociationProposal]] = {}
        for proposal in self.proposals:
            grouped.setdefault(proposal.source, []).append(proposal)
        for items in grouped.values():
            items.sort(key=lambda p: (-p.similarity, p.target))
        return grouped


def tag_surface(tag: CanonicalTag) -> str:
    """
    The text a tag is embedded as.

    The canonical name leads because it carries the concept; aliases follow because they
    are the words humans actually typed, and those are what latent space has seen. Keeping
    both means "auth" and "authentication" land near each other without an embedding of the
    word "alias".
    """
    aliases = ", ".join(rule.alias for rule in tag.aliases)
    if aliases:
        return f"{tag.name}: {aliases}"
    return tag.name


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity with an honest zero for degenerate vectors."""
    if not a or not b or len(a) != len(b):
        raise LatentError(
            f"cannot compare vectors of length {len(a)} and {len(b)}; "
            "one embedder, one dimension"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _excludes(tag: CanonicalTag) -> set:
    return {normalize_tag(name) for name in tag.excludes}


def propose_associations(
    taxonomy: TagTaxonomy,
    embeddings: Dict[str, List[float]],
    threshold: float = MIN_SIMILARITY,
    max_per_tag: int = DEFAULT_MAX_PER_TAG,
    embedder: str = "unspecified",
) -> LatentReport:
    """
    Propose association edges from latent similarity.

    Rules, all of them load bearing:

    - only canonical tags present in `embeddings` are compared; a missing vector is missing
      evidence, not a zero
    - an association already configured is skipped, so a reviewed weight is never
      overwritten by an embedding
    - an exclusion pair is never proposed, in either direction
    - below the threshold is silence, and the count of silences is reported
    - each tag keeps its strongest `max_per_tag` neighbours
    """
    if not embeddings:
        raise LatentError("no embeddings supplied; refusing to propose edges from nothing")

    by_name: Dict[str, CanonicalTag] = {
        tag.normalized_name(): tag for tag in taxonomy.tags
    }
    # accept embeddings keyed by any spelling of the tag, report keys we do not know
    vectors: Dict[str, List[float]] = {}
    for key, vector in embeddings.items():
        normalized = normalize_tag(key)
        if normalized in by_name:
            vectors[normalized] = list(vector)
    if not vectors:
        raise LatentError(
            "no embedding matched a canonical tag in this taxonomy; "
            "embed the same tags the taxonomy defines"
        )

    dimensions = {len(v) for v in vectors.values()}
    if len(dimensions) > 1:
        raise LatentError(f"mixed embedding dimensions {sorted(dimensions)}")

    names = sorted(vectors)
    all_similarities: List[float] = []
    considered: List[Tuple[float, str, str]] = []
    skipped_related = 0
    blocked = 0
    below = 0

    for i, source in enumerate(names):
        tag = by_name[source]
        existing = {normalize_tag(name) for name in tag.related}
        exclusions = _excludes(tag)
        for target in names[i + 1:]:
            other = by_name[target]
            if target in existing or source in {
                normalize_tag(name) for name in other.related
            }:
                skipped_related += 1
                continue
            if target in exclusions or source in _excludes(other):
                blocked += 1
                continue
            similarity = cosine_similarity(vectors[source], vectors[target])
            all_similarities.append(similarity)
            if similarity < threshold:
                below += 1
                continue
            considered.append((similarity, source, target))

    # strongest pairs win the per-tag budget, so a hub cannot eat every neighbour
    considered.sort(key=lambda item: (-item[0], item[1], item[2]))
    budget: Dict[str, int] = {}
    proposals: List[AssociationProposal] = []
    for similarity, source, target in considered:
        if budget.get(source, 0) >= max_per_tag or budget.get(target, 0) >= max_per_tag:
            continue
        budget[source] = budget.get(source, 0) + 1
        budget[target] = budget.get(target, 0) + 1
        proposals.append(
            AssociationProposal(
                source=by_name[source].name,
                target=by_name[target].name,
                similarity=round(similarity, 4),
                strength=round(similarity, 3),
                tier=tier_for(similarity).value,
                embedder=embedder,
            )
        )

    proposals.sort(key=lambda p: (p.source.lower(), -p.similarity, p.target.lower()))
    ordered = sorted(all_similarities)
    median = ordered[len(ordered) // 2] if ordered else 0.0
    p90 = ordered[int(round(0.9 * (len(ordered) - 1)))] if ordered else 0.0
    return LatentReport(
        embedder=embedder,
        dimension=dimensions.pop(),
        threshold=threshold,
        tags_embedded=len(vectors),
        proposals=proposals,
        skipped_already_related=skipped_related,
        blocked_by_exclusion=blocked,
        below_threshold=below,
        pairs_compared=len(all_similarities),
        similarity_median=round(median, 4),
        similarity_p90=round(p90, 4),
    )


def apply_proposals(taxonomy: TagTaxonomy, report: LatentReport) -> TagTaxonomy:
    """
    Accept every proposal, returning a new taxonomy.

    Called only after a human has read the YAML, which is why it takes the whole report: an
    accepted edge carries its similarity as the strength, and the taxonomy keeps no memory
    of the embedding afterwards. If a proposal would contradict a configured weight, the
    configured weight wins.
    """
    by_name = {tag.normalized_name(): tag for tag in taxonomy.tags}
    for proposal in report.proposals:
        source = by_name.get(normalize_tag(proposal.source))
        target = by_name.get(normalize_tag(proposal.target))
        if source is None or target is None:
            continue
        if (
            normalize_tag(target.name) in _excludes(source)
            or normalize_tag(source.name) in _excludes(target)
        ):
            continue
        source.related.setdefault(target.name, proposal.strength)
        target.related.setdefault(source.name, proposal.strength)
    return taxonomy


def proposals_to_yaml(report: LatentReport) -> str:
    """Serialize proposals for review, provenance first."""
    import yaml

    payload = {
        "provenance": {
            "embedder": report.embedder,
            "dimension": report.dimension,
            "threshold": report.threshold,
            "tags_embedded": report.tags_embedded,
            "note": (
                "latent proposals, not graph edges. review, then run with --apply "
                "to write the accepted strengths into the taxonomy."
            ),
        },
        "counts": {
            "proposals": len(report.proposals),
            "pairs_compared": report.pairs_compared,
            "similarity_median": report.similarity_median,
            "similarity_p90": report.similarity_p90,
            "skipped_already_related": report.skipped_already_related,
            "blocked_by_exclusion": report.blocked_by_exclusion,
            "below_threshold": report.below_threshold,
        },
        "proposals": [
            {
                "source": proposal.source,
                "target": proposal.target,
                "similarity": proposal.similarity,
                "strength": proposal.strength,
                "tier": proposal.tier,
            }
            for proposal in report.proposals
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def parse_embedding_response(payload: object, tags: Sequence[str]) -> Dict[str, List[float]]:
    """
    Read an OpenAI-shaped embeddings response back into tag -> vector.

    Index order is the contract: response item i belongs to tags[i]. An unknown shape is an
    error, because silently zipping a reordered list would attach the wrong vector to a tag.
    """
    if not isinstance(payload, dict):
        raise LatentError("embedding response was not an object")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(tags):
        raise LatentError(
            f"embedding response had {len(data) if isinstance(data, list) else 'no'} items "
            f"for {len(tags)} tags"
        )
    vectors: Dict[str, List[float]] = {}
    for index, item in enumerate(data):
        vector = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(vector, list) or not vector:
            raise LatentError(f"embedding {index} was empty")
        vectors[tags[index]] = [float(value) for value in vector]
    return vectors


def embedding_texts(taxonomy: TagTaxonomy) -> List[str]:
    """The surface text for every canonical tag, in a stable order."""
    return [tag_surface(tag) for tag in taxonomy.tags]


def embed_taxonomy(
    embedder,
    taxonomy: TagTaxonomy,
    batch_size: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Embed every canonical tag through a build-time embedder.

    The embedder is anything with `embed(texts) -> List[List[float]]` and a `name`. Batching
    is the embedder's default unless a size is given, because the provider owns its limits.
    """
    texts = embedding_texts(taxonomy)
    if not texts:
        raise LatentError("taxonomy has no canonical tags to embed")
    vectors: List[List[float]] = []
    step = batch_size or len(texts)
    for start in range(0, len(texts), step):
        chunk = texts[start:start + step]
        out = embedder.embed(chunk)
        if len(out) != len(chunk):
            raise LatentError(
                f"embedder returned {len(out)} vectors for {len(chunk)} tags"
            )
        vectors.extend(out)
    return {
        tag.normalized_name(): vector
        for tag, vector in zip(taxonomy.tags, vectors)
    }
