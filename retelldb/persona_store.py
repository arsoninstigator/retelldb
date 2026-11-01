"""
Persona profile storage.
A persona's voice profile is stored as a single node in their own memory graph, so
the profile travels with the memory instead of living in a separate config store.
"""

from typing import Optional
from retelldb.models.schema import (
    NodeType,
    GraphNode,
    generate_node_id,
    normalize_namespace,
)
from retelldb.models.persona import PersonaProfile
from retelldb.models.query import IngestResponse
import json
import time

PROFILE_LOCATOR = "profile"
PROFILE_TITLE = "persona voice profile"


def _profile_node_id(namespace: str) -> str:
    return generate_node_id(
        namespace,
        None,
        PROFILE_TITLE,
        NodeType.PROFILE.value,
        PROFILE_LOCATOR,
    )


def save_profile(graph_store, config, profile: PersonaProfile) -> IngestResponse:
    """Upsert a persona's voice profile into their memory graph."""
    t_start = time.perf_counter()
    ns = normalize_namespace(profile.persona_id)
    graph = config.storage.graph_for(ns)

    node = GraphNode(
        id=_profile_node_id(ns),
        title=profile.display_name or profile.persona_id,
        type=NodeType.PROFILE,
        content=profile.model_dump_json(),
        namespace=ns,
        tags=["profile"],
        locator=PROFILE_LOCATOR,
    )
    graph_store.add_node(graph, node)

    return IngestResponse(
        success=True,
        nodes_created=1,
        edges_created=0,
        tags_created=0,
        graph=graph,
        execution_time_ms=round((time.perf_counter() - t_start) * 1000, 2),
    )


def load_profile(graph_store, config, persona_id: str) -> Optional[PersonaProfile]:
    """Load a persona's voice profile from their memory graph, if one exists."""
    ns = normalize_namespace(persona_id)
    graph = config.storage.graph_for(ns)
    node = graph_store.get_node(graph, _profile_node_id(ns))
    if node is None or node.type != NodeType.PROFILE:
        return None
    return PersonaProfile.model_validate_json(node.content)
