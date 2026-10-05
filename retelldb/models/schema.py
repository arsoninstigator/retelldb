"""
Schema definitions for RetellDB: node types, edge types, and graph representations.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid

# Namespace UUID for deterministic UUIDv5 generation across documents and nodes
RETELL_NAMESPACE = uuid.UUID("a2839210-61b4-4b57-a3d8-e160e1d89b33")


def generate_node_id(
    audience: str,
    source_uri: Optional[str],
    doc_title: str,
    node_type: str,
    locator: str
) -> str:
    """
    Generate a deterministic UUIDv5 for a node based on document identity and locator.
    Enables idempotent upserts and re-ingestion.
    """
    doc_identity = source_uri.strip().lower() if source_uri else doc_title.strip().lower()
    raw_key = f"{audience}:{doc_identity}:{node_type}:{locator.strip().lower()}"
    return str(uuid.uuid5(RETELL_NAMESPACE, raw_key))


def generate_edge_id(source_id: str, target_id: str, edge_type: str) -> str:
    """Generate a deterministic UUIDv5 for a directed edge."""
    raw_key = f"{source_id}->{edge_type}->{target_id}"
    return str(uuid.uuid5(RETELL_NAMESPACE, raw_key))


class NodeType(str, Enum):
    """Supported node types in the documentation graph."""
    CONCEPT = "concept"
    API_ENDPOINT = "api_endpoint"
    SDK_METHOD = "sdk_method"
    EXAMPLE = "example"
    GUIDE = "guide"
    ERROR = "error"
    TAG = "tag"


class EdgeType(str, Enum):
    """Supported edge types connecting documentation nodes."""
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    REPLACES = "replaces"
    RELATED = "related"
    TAGGED_AS = "tagged_as"
    EXAMPLE_OF = "example_of"


class AudienceType(str, Enum):
    """Target audience graph namespace."""
    DEVELOPER = "developer"
    USER = "user"


class GraphNode(BaseModel):
    """A node in the RetellDB documentation graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    type: NodeType
    content: str = ""
    audience: AudienceType = AudienceType.DEVELOPER
    tags: List[str] = Field(default_factory=list)
    source_uri: Optional[str] = None
    locator: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_cypher_props(self, max_content_length: Optional[int] = None) -> Dict[str, Any]:
        """Convert node properties to a dictionary safe for Cypher insertion."""
        content_val = self.content
        if max_content_length is not None and len(content_val) > max_content_length:
            content_val = content_val[:max_content_length]

        props = {
            "id": self.id,
            "title": self.title,
            "type": self.type.value,
            "content": content_val,
            "audience": self.audience.value,
            "tags": self.tags,
            "source_uri": self.source_uri or "",
            "locator": self.locator or "",
        }
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                props[f"meta_{k}"] = v
        return props


class GraphEdge(BaseModel):
    """A directed edge in the RetellDB documentation graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_cypher_props(self) -> Dict[str, Any]:
        props = {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "weight": self.weight,
        }
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                props[f"meta_{k}"] = v
        return props
