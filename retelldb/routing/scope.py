"""
Platform scope: carry over or drop, per persona and per platform
(anne, oct 29 2026).

retrieval used to be scoped by persona and namespace only. but prixie
lives on many surfaces now: a zoom meeting, a discord dm, a whatsapp
call, a livestream. the same persona's memory must NOT pour across all
of them freely, and it must not treat every join as a brand new person
either. a scope is the pair that decides both:

  MemoryScope(persona_id, platform, connection_id)

and the decision is one function, fail closed:

  carry_decision(source, target, node_tags, shared=...)

the rules, in order (first match wins):
1. unknown platform on either side            -> drop. a surface we do
   not model cannot be reasoned about, so nothing crosses it.
2. another persona (non-shared node)          -> drop. student-you does
   not leak into founder-you.
3. a node anchored to a connection, and the target session is anchored
   to NONE                               -> drop. a person's context
   does not surface in an unanchored session (a meeting full of
   strangers).
4. a node anchored to one connection, the target to a DIFFERENT one
   -> drop. one person's context never crosses into another's session.
5. meeting surfaces -> public social surfaces -> drop. what was said
   in a meeting never surfaces on a stream or a post.
6. sensitivity:private                        -> carry only into a
   PRIVATE surface, for the same connection. everything else drops.
7. otherwise                                  -> carry: same persona,
   anchored to the same person or global.

writes get the same treatment in reverse: provenance(scope) stamps
platform:<p> and connection:<id> tags on every ingested node, so the
rules can read them back. a node with no platform tag is global/older
memory and carries per rule 7.

the point anne made: predefined handles make a reconnect a
CONTINUATION (same connection id -> context carries), while the rules
above keep that continuity from becoming a leak.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from retelldb.models.connections import platform_class

CONNECTION_TAG_PREFIX = "connection:"
PLATFORM_TAG_PREFIX = "platform:"
PRIVATE_SENSITIVITY_TAG = "sensitivity:private"


@dataclass
class MemoryScope:
    """Who is running, on what surface, in front of whom."""

    persona_id: Optional[str] = None
    platform: Optional[str] = None
    connection_id: Optional[str] = None

    @property
    def platform_class(self) -> str:
        return platform_class(self.platform)

    def provenance_tags(self) -> List[str]:
        """The tags stamped on writes made in this scope."""
        tags: List[str] = []
        if self.platform:
            tags.append(f"{PLATFORM_TAG_PREFIX}{self.platform.strip().lower()}")
        if self.connection_id:
            tags.append(f"{CONNECTION_TAG_PREFIX}{self.connection_id}")
        return tags

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "persona_id": self.persona_id,
            "platform": self.platform,
            "connection_id": self.connection_id,
        }


@dataclass
class CarryDecision:
    carry: bool
    reason: str


def node_platform(tags: Iterable[str]) -> Optional[str]:
    for t in tags or []:
        if t.startswith(PLATFORM_TAG_PREFIX):
            return t[len(PLATFORM_TAG_PREFIX):] or None
    return None


def node_connection(tags: Iterable[str]) -> Optional[str]:
    for t in tags or []:
        if t.startswith(CONNECTION_TAG_PREFIX):
            return t[len(CONNECTION_TAG_PREFIX):] or None
    return None


def node_is_private(tags: Iterable[str]) -> bool:
    return PRIVATE_SENSITIVITY_TAG in (tags or [])


def carry_decision(
    source: MemoryScope,
    target: MemoryScope,
    node_tags: Iterable[str] = (),
    shared: bool = False,
) -> CarryDecision:
    """May a memory from `source` be used in a session on `target`?"""
    tags = list(node_tags or ())

    # 1. unknown surface: nothing crosses a surface we cannot reason about
    if source.platform_class == "unknown" or target.platform_class == "unknown":
        return CarryDecision(False, "unknown surface on one side, nothing crosses it")

    # 2. another persona's memory (unless it lives in the shared graph)
    if not shared and (source.persona_id or None) != (target.persona_id or None):
        return CarryDecision(False, "another persona's memory")

    node_conn = node_connection(tags)

    # 3. an anchored person's context in an unanchored session
    if node_conn and not target.connection_id:
        return CarryDecision(
            False,
            "a person's context does not surface in an unanchored session",
        )

    # 4. one person's context never crosses into another person's session
    if node_conn and target.connection_id and node_conn != target.connection_id:
        return CarryDecision(False, "another person's context never crosses")

    # 5. meeting -> public social: what was said in the room stays there
    if source.platform_class == "meeting" and target.platform_class == "public_social":
        return CarryDecision(False, "meeting captures never surface on public surfaces")

    # 6. explicitly private memory: only inside a private surface, same person
    if node_is_private(tags):
        if target.platform_class == "private" and target.connection_id and (
            node_conn == target.connection_id
        ):
            return CarryDecision(True, "private memory, private surface, same person")
        return CarryDecision(False, "private memory stays in a private surface with its person")

    # 7. carry: same persona, anchored to the same person or global memory
    return CarryDecision(True, "same persona, anchored or global")


def filter_nodes(
    scored_nodes: Iterable,
    source: MemoryScope,
    target: MemoryScope,
    shared_ids: Optional[set] = None,
) -> Tuple[List, List[Dict[str, str]]]:
    """Split retrieved nodes into (kept, dropped-with-reasons).

    `scored_nodes` are objects with a `.node` having `.tags` and
    `.namespace` (retelldb's ScoredNode shape). nodes in the shared graph
    pass `shared=True` to rule 2.
    """
    shared_ids = shared_ids or set()
    kept: List = []
    dropped: List[Dict[str, str]] = []
    for sn in scored_nodes:
        node = sn.node
        is_shared = node.namespace in shared_ids
        # a node's own platform tag is its true origin (where it was
        # written); the caller's scope is the fallback for untagged
        # global memory
        node_source = MemoryScope(
            persona_id=source.persona_id,
            platform=node_platform(node.tags) or source.platform,
            connection_id=node_connection(node.tags) or source.connection_id,
        )
        decision = carry_decision(node_source, target, node.tags, shared=is_shared)
        if decision.carry:
            kept.append(sn)
        else:
            dropped.append(
                {
                    "node_id": node.id,
                    "title": node.title,
                    "reason": decision.reason,
                    "node_platform": node_platform(node.tags) or "",
                    "node_connection": node_connection(node.tags) or "",
                }
            )
    return kept, dropped


def scoped_namespaces(persona_id: Optional[str]) -> List[str]:
    """The graphs a session may read: its persona, plus shared."""
    from retelldb.models.schema import SHARED_NAMESPACE

    if persona_id:
        return [persona_id, SHARED_NAMESPACE]
    return [SHARED_NAMESPACE]


def provenance(scope: MemoryScope) -> Dict[str, object]:
    """Where a write happened: stamped on nodes and receipts alike."""
    return {
        "tags": scope.provenance_tags(),
        "scope": scope.as_dict(),
    }
