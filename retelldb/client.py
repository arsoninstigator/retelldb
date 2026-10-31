"""
RetellDB Client SDK.
Supports both embedded in-process execution and remote HTTP hosted API communication.
"""

from typing import Any, Dict, List, Optional, Union
import httpx
from retelldb.models.schema import AudienceType
from retelldb.models.query import (
    QueryRequest,
    QueryResponse,
    SubGraphResult,
    IngestRequest,
    IngestResponse,
    IngestDocumentFormat,
    GraphStatusResponse,
)
from retelldb.config import RetellDBConfig, default_config
from retelldb.storage.memory_store import MemoryGraphStore, MemoryVectorStore
from retelldb.storage.falkordb_store import FalkorDBStore
from retelldb.storage.pgvector_store import PGVectorStore
from retelldb.ingestion.pipeline import IngestionPipeline
from retelldb.routing.engine import RetrievalEngine
from retelldb.wrangling.citation import SourceVerifier
from retelldb.wrangling.taxonomy import TagTaxonomy


class RetellDB:
    """Client for querying and ingesting documents into RetellDB."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        config: Optional[RetellDBConfig] = None,
        use_memory: bool = False,
        taxonomy: Optional["TagTaxonomy"] = None,
        source_verifier: Optional["SourceVerifier"] = None,
    ):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.config = config or default_config
        self.taxonomy = taxonomy
        self.source_verifier = source_verifier

        if self.base_url:
            self._http_client = httpx.Client(base_url=self.base_url, timeout=30.0)
            self._embedded = False
        else:
            self._embedded = True
            # Embedded initialization
            if use_memory or self.config.storage.backend == "memory":
                self.graph_store = MemoryGraphStore()
                self.vector_store = MemoryVectorStore()
            else:
                self.graph_store = FalkorDBStore(self.config.storage)
                self.vector_store = PGVectorStore(self.config.storage)

            self.pipeline = IngestionPipeline(
                self.graph_store,
                self.vector_store,
                self.config,
                taxonomy=self.taxonomy,
            )
            self.engine = RetrievalEngine(
                self.graph_store,
                self.vector_store,
                self.config,
                taxonomy=self.taxonomy,
                source_verifier=self.source_verifier,
            )

    def query(
        self,
        query: str,
        audience: Optional[Union[str, AudienceType]] = None,
        max_hops: int = 3,
        top_k: int = 5,
        include_edges: bool = True,
        persona: Optional[str] = None,
        allow_shared: bool = True,
        tag_confidence_threshold: float = 0.6,
        min_edge_weight: Optional[float] = None,
        include_blocked: bool = True,
        include_citations: bool = True,
    ) -> SubGraphResult:
        """Query documentation graphs via routing layer."""
        aud_enum = AudienceType(audience) if isinstance(audience, str) else audience
        req = QueryRequest(
            query=query,
            audience=aud_enum,
            max_hops=max_hops,
            top_k=top_k,
            include_edges=include_edges,
            persona=persona,
            allow_shared=allow_shared,
            tag_confidence_threshold=tag_confidence_threshold,
            min_edge_weight=min_edge_weight,
            include_blocked=include_blocked,
            include_citations=include_citations,
        )

        if not self._embedded:
            resp = self._http_client.post("/v1/query", json=req.model_dump())
            resp.raise_for_status()
            data = resp.json()
            return SubGraphResult.model_validate(data["result"])

        return self.engine.retrieve(req)

    def ingest(
        self,
        content: str,
        format: Union[str, IngestDocumentFormat] = "markdown",
        audience: Union[str, AudienceType] = "developer",
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        personas: Optional[List[str]] = None,
        sensitivity: str = "shared",
    ) -> IngestResponse:
        """Ingest document content into the property graph and vector stores."""
        fmt_enum = IngestDocumentFormat(format) if isinstance(format, str) else format
        aud_enum = AudienceType(audience) if isinstance(audience, str) else audience

        req = IngestRequest(
            content=content,
            format=fmt_enum,
            audience=aud_enum,
            title=title,
            source_uri=source_uri,
            metadata=metadata or {},
            personas=personas or [],
            sensitivity=sensitivity,
        )

        if not self._embedded:
            resp = self._http_client.post("/v1/ingest", json=req.model_dump())
            resp.raise_for_status()
            return IngestResponse.model_validate(resp.json())

        return self.pipeline.ingest(
            content=req.content,
            format=req.format,
            audience=req.audience,
            title=req.title,
            source_uri=req.source_uri,
            metadata=req.metadata,
            personas=req.personas,
            sensitivity=req.sensitivity,
        )

    def status(self) -> GraphStatusResponse:
        """Check status of dev and user documentation graphs."""
        if not self._embedded:
            resp = self._http_client.get("/v1/status")
            resp.raise_for_status()
            return GraphStatusResponse.model_validate(resp.json())

        dev_status = self.graph_store.get_status(self.config.storage.dev_graph_name)
        user_status = self.graph_store.get_status(self.config.storage.user_graph_name)
        embed_status = self.pipeline.embedder.get_status()
        return GraphStatusResponse(
            dev_graph=dev_status,
            user_graph=user_status,
            embedding_subsystem=embed_status
        )
