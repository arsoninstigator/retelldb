"""
Confidence tiers and structured doubt.

A retrieval engine that only returns a float invites an LLM to stretch a weak match
into a confident answer. This module turns the numbers into a tier and an explicit
account of what dragged the score down, so the agent on top can hedge, ask, or
refuse instead of filling the gap.

Dampeners multiply (weak signals compound) but each is floored, so a single weak
component cannot collapse an otherwise solid match to zero. The floor is
configurable; the default is 0.45.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

DEFAULT_DAMPENER_FLOOR = 0.45

# Tier boundaries are inclusive lower bounds.
TIER_BANDS = (
    (0.75, "high"),
    (0.45, "uncertain"),
    (0.25, "needs_clarification"),
    (0.0, "unavailable"),
)


class ConfidenceTier(str, Enum):
    """Behavioral tier derived from the total confidence score."""

    HIGH = "high"
    UNCERTAIN = "uncertain"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNAVAILABLE = "unavailable"


# What an agent is expected to do at each tier.
TIER_BEHAVIOR = {
    ConfidenceTier.HIGH: "answer directly, no hedging",
    ConfidenceTier.UNCERTAIN: "answer, but flag the weak signal that limited it",
    ConfidenceTier.NEEDS_CLARIFICATION: "ask a clarifying question instead of guessing",
    ConfidenceTier.UNAVAILABLE: "refuse and offer the next best action",
}

COMPONENT_ORDER = (
    "hierarchy_decay",
    "path_weight",
    "persona_relevance",
    "source_confidence",
)

# Which recovery action addresses which weak component.
COMPONENT_ACTIONS = {
    "hierarchy_decay": "expand_to_parent_tags",
    "path_weight": "deepen_traversal",
    "persona_relevance": "check_other_perspective",
    "source_confidence": "prefer_primary_source",
}


def tier_for(score: float) -> ConfidenceTier:
    """Map a score to its behavioral tier."""
    clamped = max(0.0, min(1.0, float(score)))
    for lower, tier in TIER_BANDS:
        if clamped >= lower:
            return ConfidenceTier(tier)
    return ConfidenceTier.UNAVAILABLE


class ConfidenceBreakdown(BaseModel):
    """Total confidence plus the components that produced it."""

    total: float
    tier: ConfidenceTier
    components: Dict[str, float] = Field(default_factory=dict)
    dominant_dampener: Optional[str] = None
    next_actions: List[str] = Field(default_factory=list)
    behavior: str = ""

    def is_answerable(self) -> bool:
        return self.tier in (ConfidenceTier.HIGH, ConfidenceTier.UNCERTAIN)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_breakdown(
    base_signal: float,
    hierarchy_decay: float = 1.0,
    path_weight: float = 1.0,
    persona_relevance: float = 1.0,
    source_confidence: float = 1.0,
    floor: float = DEFAULT_DAMPENER_FLOOR,
    alternative_personas: Optional[List[str]] = None,
) -> ConfidenceBreakdown:
    """
    Combine the retrieval signal with its dampeners into a tiered verdict.

    `base_signal` is what matched (tag confidence or vector similarity). Every other
    argument is a dampener in [0, 1] where 1 means "no penalty".
    """
    components = {
        "hierarchy_decay": _clamp(hierarchy_decay),
        "path_weight": _clamp(path_weight),
        "persona_relevance": _clamp(persona_relevance),
        "source_confidence": _clamp(source_confidence),
    }

    total = _clamp(base_signal)
    for name in COMPONENT_ORDER:
        damped = floor + (1.0 - floor) * components[name]
        total *= damped
    total = _clamp(total)

    tier = tier_for(total)

    dominant: Optional[str] = None
    weakest = -1.0
    for name in COMPONENT_ORDER:
        weakness = 1.0 - components[name]
        if weakness > weakest + 1e-9:
            weakest = weakness
            dominant = name
    if weakest <= 1e-9:
        dominant = None

    actions: List[str] = []
    if tier == ConfidenceTier.HIGH:
        actions.append("answer_directly")
    elif tier == ConfidenceTier.UNCERTAIN:
        actions.append("answer_with_caveat")
    elif tier == ConfidenceTier.NEEDS_CLARIFICATION:
        actions.append("ask_clarifying_question")
    else:
        actions.append("state_unavailable")

    if dominant and tier != ConfidenceTier.HIGH:
        follow_up = COMPONENT_ACTIONS[dominant]
        if follow_up not in actions:
            actions.append(follow_up)

    if tier in (ConfidenceTier.NEEDS_CLARIFICATION, ConfidenceTier.UNAVAILABLE):
        for persona in sorted(set(alternative_personas or [])):
            actions.append(f"switch_persona:{persona}")
        if alternative_personas:
            actions.append("check_shared_archive")

    return ConfidenceBreakdown(
        total=round(total, 4),
        tier=tier,
        components={k: round(v, 4) for k, v in components.items()},
        dominant_dampener=dominant,
        next_actions=actions,
        behavior=TIER_BEHAVIOR[tier],
    )
