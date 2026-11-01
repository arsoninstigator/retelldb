"""
Schema definitions for RetellDB: node types, edge types, and graph representations.
"""

import ast
import json
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid

# Namespace UUID for deterministic UUIDv5 generation across documents and nodes
RETELL_NAMESPACE = uuid.UUID("a2839210-61b4-4b57-a3d8-e160e1d89b33")

# the namespace that holds memory shared across all personas
SHARED_NAMESPACE = "shared"


def normalize_namespace(namespace: Optional[str]) -> str:
    """Sanitize a persona id into a safe graph namespace token."""
    if not namespace:
        return SHARED_NAMESPACE
    token = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in namespace.strip().lower())
    return token or SHARED_NAMESPACE


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
    # persona memory nodes (the memorium model). docs and memory share one graph shape,
    # so a retrieval engine and its provenance work the same for both.
    FACT = "fact"                 # something the persona knows to be true
    PREFERENCE = "preference"     # how the persona likes things done
    RELATIONSHIP = "relationship" # who the persona knows and how
    EVENT = "event"               # something that happened, with a when
    SKILL = "skill"               # something the persona can do
    ENTITY = "entity"             # a named thing other memories reference
    PROFILE = "profile"           # the persona's own voice profile (one per persona graph)


class EdgeType(str, Enum):
    """Supported edge types connecting documentation nodes."""
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    REPLACES = "replaces"
    RELATED = "related"
    TAGGED_AS = "tagged_as"
    EXAMPLE_OF = "example_of"
    # tag wrangling
    ALIAS_OF = "alias_of"    # alias tag -> the canonical tag it resolves to
    PARENT_OF = "parent_of"  # child tag -> the broader tag that contains it
    # a negative association: traversal never walks through this edge
    BLOCKS = "blocks"


class AudienceType(str, Enum):
    """Target audience graph namespace."""
    DEVELOPER = "developer"
    USER = "user"


class SourceRef(BaseModel):
    """
    Where a node actually came from: a file, a line range inside it, and what kind
    of content sits there. This is what makes a citation checkable instead of
    decorative.
    """
    uri: Optional[str] = None
    path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    content_type: str = "text"
    excerpt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_type": self.content_type,
            "excerpt": self.excerpt,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SourceRef":
        if not isinstance(data, dict):
            return cls()
        return cls(
            uri=data.get("uri"),
            path=data.get("path"),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            content_type=data.get("content_type") or "text",
            excerpt=data.get("excerpt"),
        )


class GraphNode(BaseModel):
    """A node in the RetellDB documentation graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    type: NodeType
    content: str = ""
    audience: AudienceType = AudienceType.DEVELOPER
    # which memory namespace this node belongs to ("shared" or a persona id). does not
    # change docs behavior; persona memory uses it to keep one head per graph.
    namespace: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source_uri: Optional[str] = None
    locator: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    # where this node lives in the source material (path, line range, content type)
    sources: List[SourceRef] = Field(default_factory=list)
    # perspectives that may read this node; empty means shared/unscoped
    personas: List[str] = Field(default_factory=list)
    # "shared" or "private": private nodes are only visible to their personas
    sensitivity: str = "shared"

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
            "personas": list(self.personas),
            "sensitivity": self.sensitivity or "shared",
            "sources_json": json.dumps([s.to_dict() for s in self.sources]),
        }
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                props[f"meta_{k}"] = v
        return props

    @staticmethod
    def _coerce_list(raw: Any) -> List[str]:
        """Coerce a Cypher property that should be a list of strings."""
        if isinstance(raw, list):
            return [str(v) for v in raw]
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
            return [raw]
        return []

    @classmethod
    def from_cypher_props(cls, props: Dict[str, Any]) -> "GraphNode":
        """Rebuild a node from raw backend properties, tolerating backend quirks."""
        raw_sources = props.get("sources_json")
        sources: List[SourceRef] = []
        if isinstance(raw_sources, str) and raw_sources.strip():
            try:
                parsed = json.loads(raw_sources)
                if isinstance(parsed, list):
                    sources = [SourceRef.from_dict(item) for item in parsed]
            except (ValueError, TypeError):
                sources = []

        try:
            node_type = NodeType(str(props.get("type", "concept")).lower())
        except ValueError:
            node_type = NodeType.CONCEPT
        try:
            audience = AudienceType(str(props.get("audience", "developer")).lower())
        except ValueError:
            audience = AudienceType.DEVELOPER

        return cls(
            id=props.get("id") or "",
            title=props.get("title", ""),
            type=node_type,
            content=props.get("content", ""),
            audience=audience,
            tags=cls._coerce_list(props.get("tags", [])),
            source_uri=props.get("source_uri") or None,
            locator=props.get("locator") or None,
            sources=sources,
            personas=cls._coerce_list(props.get("personas", [])),
            sensitivity=props.get("sensitivity") or "shared",
        )


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

    @staticmethod
    def _coerce_list(raw: Any) -> List[str]:
        """Coerce a Cypher property that should be a list of strings."""
        if isinstance(raw, list):
            return [str(v) for v in raw]
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
            return [raw]
        return []

    @classmethod
    def from_cypher_props(cls, props: Dict[str, Any]) -> "GraphNode":
        """Rebuild a node from raw backend properties, tolerating backend quirks."""
        raw_sources = props.get("sources_json")
        sources: List[SourceRef] = []
        if isinstance(raw_sources, str) and raw_sources.strip():
            try:
                parsed = json.loads(raw_sources)
                if isinstance(parsed, list):
                    sources = [SourceRef.from_dict(item) for item in parsed]
            except (ValueError, TypeError):
                sources = []

        try:
            node_type = NodeType(str(props.get("type", "concept")).lower())
        except ValueError:
            node_type = NodeType.CONCEPT
        try:
            audience = AudienceType(str(props.get("audience", "developer")).lower())
        except ValueError:
            audience = AudienceType.DEVELOPER

        return cls(
            id=props.get("id") or "",
            title=props.get("title", ""),
            type=node_type,
            content=props.get("content", ""),
            audience=audience,
            tags=cls._coerce_list(props.get("tags", [])),
            source_uri=props.get("source_uri") or None,
            locator=props.get("locator") or None,
            sources=sources,
            personas=cls._coerce_list(props.get("personas", [])),
            sensitivity=props.get("sensitivity") or "shared",
        )
