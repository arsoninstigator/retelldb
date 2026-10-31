"""
Query contracts and response shapes for RetellDB.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from retelldb.models.schema import AudienceType, GraphNode, GraphEdge
from retelldb.wrangling.citation import Citation
from retelldb.wrangling.confidence import ConfidenceBreakdown
from retelldb.wrangling.scope import ScopeBlock
from retelldb.wrangling.taxonomy import TagResolution


class QueryStrategy(str, Enum):
    """Retrieval strategies used by RetellDB routing layer."""
    TAG_TRAVERSAL = "tag_traversal"
    VECTOR_FALLBACK = "vector_fallback"
    HYBRID = "hybrid"


class QueryRequest(BaseModel):
    """Agent query payload shape as defined in agent.md."""
    query: str
    audience: Optional[AudienceType] = None  # None = routing layer classifies automatically
    max_hops: int = Field(default=3, ge=1, le=10)
    top_k: int = Field(default=5, ge=1, le=50)
    include_edges: bool = True
    tag_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    # perspective: which persona this query runs as. None means no declared perspective,
    # in which case persona-scoped content stays out of reach (fail-closed).
    persona: Optional[str] = None
    allow_shared: bool = True
    # report what perspective hid and where the content came from
    include_blocked: bool = True
    include_citations: bool = True
    # minimum edge weight for traversal; None uses the engine default
    min_edge_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ScoredNode(BaseModel):
    """Retrieved node with confidence and relevance scores."""
    node: GraphNode
    score: float
    confidence: float
    retrieval_source: str  # "tag_traversal", "vector_similarity", "gap_fill"


class SubGraphResult(BaseModel):
    """Structured result returned to the agent."""
    query: str
    target_graph: str
    retrieval_path: QueryStrategy
    nodes: List[ScoredNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    traversal_chains: List[List[str]] = Field(default_factory=list)
    confidence_signals: Dict[str, float] = Field(default_factory=dict)
    execution_time_ms: float = 0.0
    # tiered confidence plus the components that dampened it
    confidence: Optional[ConfidenceBreakdown] = None
    # weighted traversal traces: which path and which weights pulled each node in
    paths: List[Dict[str, Any]] = Field(default_factory=list)
    # tags the query resolved through, including aliases and their provenance
    tag_resolutions: List[TagResolution] = Field(default_factory=list)
    # nodes withheld by the active perspective, with reasons
    blocked: List[ScopeBlock] = Field(default_factory=list)
    # where the retrieved content actually came from
    citations: List[Citation] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Standard API response for queries."""
    success: bool = True
    result: SubGraphResult


class IngestDocumentFormat(str, Enum):
    """Supported document formats for ingestion."""
    MARKDOWN = "markdown"
    MDX = "mdx"
    OPENAPI = "openapi"
    ASYNCAPI = "asyncapi"
    NOTION = "notion"
    HTML = "html"


class IngestRequest(BaseModel):
    """Document ingestion payload."""
    content: str
    format: IngestDocumentFormat
    audience: AudienceType = AudienceType.DEVELOPER
    title: Optional[str] = None
    source_uri: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # perspective stamps: who may retrieve this content afterwards
    personas: List[str] = Field(default_factory=list)
    sensitivity: str = "shared"


class IngestResponse(BaseModel):
    """Document ingestion response."""
    success: bool = True
    nodes_created: int
    edges_created: int
    tags_created: int
    graph: str
    execution_time_ms: float
    # raw tags that resolved through a configured alias
    aliases_created: int = 0
    # raw tags that matched more than one canonical tag and were left alone
    ambiguous_tags: List[str] = Field(default_factory=list)


class GraphStatusResponse(BaseModel):
    """Status endpoint response for graph sizes and freshness."""
    dev_graph: Dict[str, Any]
    user_graph: Dict[str, Any]
    embedding_subsystem: Optional[Dict[str, Any]] = None
