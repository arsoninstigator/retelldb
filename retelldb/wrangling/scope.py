"""
Perspective scoping.

Perspective is a retrieval parameter, not a personality skin. A node is visible to
a persona only when the graph says so, and every decision to hide a node is recorded
with a reason so the agent can explain the gap instead of inventing around it.

The ordering is fail-closed: private content is checked first, then persona
membership, then whether shared content is in scope at all. When the scope is unset,
persona-scoped content is never visible.
"""

from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from retelldb.models.schema import GraphNode


class BlockReason(str, Enum):
    """Why a node was withheld from a perspective."""

    PRIVATE_OTHER_PERSONA = "private_other_persona"
    PERSONA_MISMATCH = "persona_mismatch"
    SCOPE_UNSET = "scope_unset"
    SHARED_EXCLUDED = "shared_excluded"


class PersonaScope(BaseModel):
    """
    The perspective a query runs as.

    `persona_id=None` means "no declared perspective": shared content is readable,
    persona-scoped content is not. That is the safe default for anonymous surfaces.
    """

    persona_id: Optional[str] = None
    allow_shared: bool = True
    # optional extra exclusion applied after visibility, e.g. tags a surface may not read
    excluded_tags: List[str] = Field(default_factory=list)

    def normalized_persona(self) -> Optional[str]:
        return self.persona_id.strip().lower() if self.persona_id else None


class ScopeBlock(BaseModel):
    """A withheld node, with the reason and the personas that would expose it."""

    node_id: str
    title: str
    reason: BlockReason
    available_personas: List[str] = Field(default_factory=list)


# relevance contribution of a node for the active perspective
RELEVANCE_EXACT = 1.0
RELEVANCE_SHARED = 0.85
RELEVANCE_BLOCKED = 0.0


def _personas(node: GraphNode) -> List[str]:
    return [p.strip().lower() for p in (node.personas or []) if str(p).strip()]


def evaluate_node(
    node: GraphNode, scope: PersonaScope
) -> Tuple[bool, Optional[BlockReason], float]:
    """
    Return (visible, block_reason, persona_relevance).

    Checked in fail-closed order: private/other-persona first, then persona
    membership, then shared scope.
    """
    persona = scope.normalized_persona()
    personas = _personas(node)
    sensitivity = (node.sensitivity or "shared").strip().lower()

    if sensitivity == "private" and personas:
        if persona is None or persona not in personas:
            return False, BlockReason.PRIVATE_OTHER_PERSONA, RELEVANCE_BLOCKED

    if personas:
        if persona is None:
            return False, BlockReason.SCOPE_UNSET, RELEVANCE_BLOCKED
        if persona not in personas:
            return False, BlockReason.PERSONA_MISMATCH, RELEVANCE_BLOCKED

    if scope.excluded_tags and set(t.lower() for t in node.tags) & set(
        t.lower() for t in scope.excluded_tags
    ):
        return False, BlockReason.SHARED_EXCLUDED, RELEVANCE_BLOCKED

    if not personas:
        if not scope.allow_shared:
            return False, BlockReason.SHARED_EXCLUDED, RELEVANCE_BLOCKED
        return True, None, RELEVANCE_SHARED

    return True, None, RELEVANCE_EXACT


def filter_nodes(
    nodes: Iterable[GraphNode], scope: Optional[PersonaScope]
) -> Tuple[List[GraphNode], List[ScopeBlock], Dict[str, float]]:
    """
    Partition nodes by visibility.

    Returns (kept, blocked, relevance_by_node_id). An absent scope is treated as
    "no declared perspective", never as "everything is visible".
    """
    active_scope = scope or PersonaScope()
    kept: List[GraphNode] = []
    blocked: List[ScopeBlock] = []
    relevance: Dict[str, float] = {}

    for node in nodes:
        visible, reason, rel = evaluate_node(node, active_scope)
        if visible:
            kept.append(node)
            relevance[node.id] = rel
            continue
        blocked.append(
            ScopeBlock(
                node_id=node.id,
                title=node.title,
                reason=reason or BlockReason.PERSONA_MISMATCH,
                available_personas=sorted(_personas(node)),
            )
        )

    return kept, blocked, relevance
