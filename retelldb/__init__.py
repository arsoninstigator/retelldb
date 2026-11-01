"""
retelldb: A graph-powered retrieval layer for agents that actually understand your docs.
"""

from retelldb.models.schema import (
    NodeType,
    EdgeType,
    AudienceType,
    GraphNode,
    GraphEdge,
)
from retelldb.models.query import (
    QueryRequest,
    QueryResponse,
    QueryStrategy,
    SubGraphResult,
    ScoredNode,
)
from retelldb.models.persona import PersonaProfile
from retelldb.persona_store import load_profile, save_profile
from retelldb.routing.persona import PersonaRouter
from retelldb.client import RetellDB

# one source of truth for the version: the installed package metadata. a source checkout
# without an install falls back to the version in this file, which a test keeps honest.
try:  # pragma: no cover - exercised by the installed package
    from importlib.metadata import version as _version

    __version__ = _version("retelldb")
except Exception:  # pragma: no cover
    __version__ = "0.2.8"
__all__ = [
    "RetellDB",
    "NodeType",
    "EdgeType",
    "AudienceType",
    "GraphNode",
    "GraphEdge",
    "QueryRequest",
    "QueryResponse",
    "QueryStrategy",
    "SubGraphResult",
    "ScoredNode",
    "PersonaProfile",
    "PersonaRouter",
    "save_profile",
    "load_profile",
]
