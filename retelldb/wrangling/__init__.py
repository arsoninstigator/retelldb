"""
Tag wrangling layer.

The wrangler is the part of retelldb that turns messy human tags into canonical
graph structure without losing what the human actually wrote:

- taxonomy: canonical tags, aliases, alias provenance, hierarchy
- traversal: weighted graph walks (edge weights, decay, blocked paths, cycles)
- scope: perspective (persona) as a retrieval parameter, fail-closed
- confidence: confidence tiers and structured doubt
- citation: source references (path, line range, content type) plus integrity checks
"""

from retelldb.wrangling.taxonomy import (
    AliasRule,
    CanonicalTag,
    ResolutionRule,
    TagResolution,
    TagTaxonomy,
)
from retelldb.wrangling.traversal import (
    TraversalResult,
    TraversedPath,
    WeightedTraverser,
)
from retelldb.wrangling.scope import (
    BlockReason,
    PersonaScope,
    ScopeBlock,
    evaluate_node,
    filter_nodes,
)
from retelldb.wrangling.confidence import (
    ConfidenceBreakdown,
    ConfidenceTier,
    build_breakdown,
    tier_for,
)
from retelldb.wrangling.citation import Citation, SourceVerifier, citations_for_node

__all__ = [
    "AliasRule",
    "CanonicalTag",
    "ResolutionRule",
    "TagResolution",
    "TagTaxonomy",
    "TraversalResult",
    "TraversedPath",
    "WeightedTraverser",
    "BlockReason",
    "PersonaScope",
    "ScopeBlock",
    "evaluate_node",
    "filter_nodes",
    "ConfidenceBreakdown",
    "ConfidenceTier",
    "build_breakdown",
    "tier_for",
    "Citation",
    "SourceVerifier",
    "citations_for_node",
]
