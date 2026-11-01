"""
Configuration settings for RetellDB.
"""

from typing import Optional, List
import os
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Storage backend configuration."""
    backend: str = Field(default_factory=lambda: os.getenv("RETELLDB_STORAGE_BACKEND", "memory"))
    
    # FalkorDB settings
    falkordb_host: str = Field(default_factory=lambda: os.getenv("FALKORDB_HOST", "localhost"))
    falkordb_port: int = Field(default_factory=lambda: int(os.getenv("FALKORDB_PORT", "6379")))
    falkordb_password: Optional[str] = Field(default_factory=lambda: os.getenv("FALKORDB_PASSWORD"))
    
    # PostgreSQL / pgvector settings
    pg_dsn: str = Field(default_factory=lambda: os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/retelldb"))
    embedding_dimension: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384")))
    
    # Graph instance names
    dev_graph_name: str = Field(default_factory=lambda: os.getenv("DEV_GRAPH_NAME", "dev_graph"))
    user_graph_name: str = Field(default_factory=lambda: os.getenv("USER_GRAPH_NAME", "user_graph"))

    # persona memory graphs: one graph per namespace, so a persona's head is a sandbox
    namespace_prefix: str = Field(default_factory=lambda: os.getenv("MEMORY_GRAPH_PREFIX", "memory"))
    shared_namespace: str = Field(default_factory=lambda: os.getenv("SHARED_NAMESPACE_NAME", "shared"))

    def graph_for(self, namespace: str) -> str:
        """Graph name for a memory namespace ('shared' -> the shared graph, else per-persona)."""
        from retelldb.models.schema import SHARED_NAMESPACE, normalize_namespace

        token = normalize_namespace(namespace)
        if token == SHARED_NAMESPACE:
            return f"{self.namespace_prefix}_{self.shared_namespace}"
        return f"{self.namespace_prefix}_{token}"


class InferenceConfig(BaseModel):
    """LiteLLM and model split configuration."""
    fast_model: str = Field(
        default_factory=lambda: os.getenv(
            "RETELLDB_FAST_MODEL", 
            "featherless/meta-llama/Llama-3.2-3B-Instruct"
        )
    )
    strong_model: str = Field(
        default_factory=lambda: os.getenv(
            "RETELLDB_STRONG_MODEL", 
            "featherless/meta-llama/Llama-3.3-70B-Instruct"
        )
    )
    # Embedding model configuration
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "RETELLDB_EMBEDDING_MODEL", 
            "text-embedding-3-small" if os.getenv("OPENAI_API_KEY") else "featherless/BAAI/bge-small-en-v1.5"
        )
    )
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("FEATHERLESS_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    embedding_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or os.getenv("FEATHERLESS_API_KEY")
    )
    # Maximum characters of node content passed to LLM during entity typing and tag extraction prompts.
    # Tradeoff: higher values preserve richer context for intricate API schemas, parameter tables,
    # and extensive walkthroughs, at the expense of higher token overhead and latency during high-volume ingestion.
    node_content_llm_max_chars: int = Field(
        default_factory=lambda: int(os.getenv("RETELLDB_NODE_CONTENT_LLM_MAX_CHARS", "1000"))
    )


class WranglingConfig(BaseModel):
    """Tag wrangling behaviour: canonical tags, walk weights, confidence tiers."""
    # path to a JSON/YAML tag taxonomy; empty means no aliasing and no hierarchy
    taxonomy_path: Optional[str] = Field(default_factory=lambda: os.getenv("RETELLDB_TAXONOMY_PATH"))
    # edges lighter than this are never walked
    min_edge_weight: float = Field(
        default_factory=lambda: float(os.getenv("RETELLDB_MIN_EDGE_WEIGHT", "0.05"))
    )
    # per-component dampener floor used when building confidence
    dampener_floor: float = Field(
        default_factory=lambda: float(os.getenv("RETELLDB_DAMPENER_FLOOR", "0.45"))
    )
    # hop distance decay applied to traversal hits
    hop_decay: float = Field(
        default_factory=lambda: float(os.getenv("RETELLDB_HOP_DECAY", "0.85"))
    )


class RetellDBConfig(BaseModel):
    """Global RetellDB config."""
    storage: StorageConfig = Field(default_factory=StorageConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    wrangling: WranglingConfig = Field(default_factory=WranglingConfig)
    taxonomy_path: Optional[str] = Field(default_factory=lambda: os.getenv("RETELLDB_TAXONOMY_PATH"))
    host: str = Field(default_factory=lambda: os.getenv("RETELLDB_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("RETELLDB_PORT", "8000")))
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("RETELLDB_API_KEY"))


# Default global instance
default_config = RetellDBConfig()
