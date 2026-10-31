"""
FastAPI application for RetellDB hosted API.
"""

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from retelldb.models.query import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    GraphStatusResponse,
)
from retelldb.config import RetellDBConfig, default_config
from retelldb.storage.memory_store import MemoryGraphStore, MemoryVectorStore
from retelldb.storage.falkordb_store import FalkorDBStore
from retelldb.storage.pgvector_store import PGVectorStore
from retelldb.ingestion.pipeline import IngestionPipeline
from retelldb.routing.engine import RetrievalEngine

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(config: RetellDBConfig = default_config) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="RetellDB API",
        version="0.1.0",
        description="Graph-powered retrieval layer for agents that actually understand your docs."
    )

    # Valid CORS specification: allow_credentials cannot be True when allow_origins=["*"]
    is_wildcard = "*" in config.allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=not is_wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize storage & engines
    if config.storage.backend == "memory":
        graph_store = MemoryGraphStore()
        vector_store = MemoryVectorStore()
    else:
        graph_store = FalkorDBStore(config.storage)
        vector_store = PGVectorStore(config.storage)

    pipeline = IngestionPipeline(graph_store, vector_store, config)
    engine = RetrievalEngine(graph_store, vector_store, config)

    def verify_api_key(key: Optional[str] = Security(api_key_header)):
        if config.api_key:
            if not key or key != config.api_key:
                raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
        return key

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": "retelldb",
            "storage_backend": config.storage.backend,
            "embedding_mode": pipeline.embedder.provider_info
        }

    @app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
    def query_docs(request: QueryRequest):
        try:
            result = engine.retrieve(request)
            return QueryResponse(success=True, result=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
    def ingest_docs(request: IngestRequest):
        try:
            resp = pipeline.ingest(
                content=request.content,
                format=request.format,
                audience=request.audience,
                title=request.title,
                source_uri=request.source_uri,
                metadata=request.metadata,
                personas=request.personas,
                sensitivity=request.sensitivity,
            )
            return resp
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/status", response_model=GraphStatusResponse, dependencies=[Depends(verify_api_key)])
    def graph_status():
        try:
            dev_status = graph_store.get_status(config.storage.dev_graph_name)
            user_status = graph_store.get_status(config.storage.user_graph_name)
            embed_status = pipeline.embedder.get_status()
            return GraphStatusResponse(
                dev_graph=dev_status,
                user_graph=user_status,
                embedding_subsystem=embed_status
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()
