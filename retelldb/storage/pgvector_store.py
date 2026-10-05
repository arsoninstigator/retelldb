"""
pgvector storage adapter for vector similarity fallback and associative proximity search.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from retelldb.storage.base import BaseVectorStore
from retelldb.storage.memory_store import MemoryVectorStore
from retelldb.config import StorageConfig, default_config

try:
    from psycopg.types.json import Jsonb
except ImportError:
    Jsonb = None  # psycopg not installed — fallback path won't need it

logger = logging.getLogger("retelldb.pgvector")


class PGVectorStore(BaseVectorStore):
    """PostgreSQL pgvector adapter for cosine similarity search over contextual embeddings."""

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or default_config.storage
        self._connected = False
        self._conn = None
        self._memory_fallback = MemoryVectorStore()

        try:
            import psycopg
            from pgvector.psycopg import register_vector
            self._conn = psycopg.connect(self.config.pg_dsn, autocommit=True)
            register_vector(self._conn)
            self._init_db()
            self._connected = True
            logger.info("Connected to pgvector successfully.")
        except Exception as e:
            logger.warning(f"Could not connect to pgvector ({e}). Using in-memory vector store fallback.")
            self._connected = False

    def _init_db(self):
        # psycopg3 does not support multiple statements in a single execute call;
        # split every DDL statement into its own cursor.execute.
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS retell_embeddings (
                    id VARCHAR(255) PRIMARY KEY,
                    graph_name VARCHAR(100) NOT NULL,
                    node_id VARCHAR(255) NOT NULL,
                    embedding vector({self.config.embedding_dimension}),
                    metadata JSONB
                );
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_retell_graph "
                "ON retell_embeddings(graph_name);"
            )

    def add_embedding(
        self, 
        graph_name: str, 
        node_id: str, 
        embedding: List[float], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self._memory_fallback.add_embedding(graph_name, node_id, embedding, metadata)
        if not self._connected:
            return

        try:
            row_id = f"{graph_name}:{node_id}"
            # Use psycopg Jsonb adapter so the driver knows to cast as JSONB;
            # fall back to a plain json string if the import failed.
            meta_val = Jsonb(metadata or {}) if Jsonb is not None else json.dumps(metadata or {})
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO retell_embeddings (id, graph_name, node_id, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                    """,
                    (row_id, graph_name, node_id, embedding, meta_val)
                )
        except Exception as e:
            logger.error(f"pgvector add_embedding failed: {e}")

    def similarity_search(
        self, 
        graph_name: str, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        if not self._connected:
            return self._memory_fallback.similarity_search(graph_name, query_embedding, top_k)

        try:
            with self._conn.cursor() as cur:
                # 1 - (embedding <=> query) gives cosine similarity
                cur.execute(
                    """
                    SELECT node_id, (1 - (embedding <=> %s::vector)) as score
                    FROM retell_embeddings
                    WHERE graph_name = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_embedding, graph_name, query_embedding, top_k)
                )
                rows = cur.fetchall()
                return [(r[0], max(0.0, float(r[1]))) for r in rows]
        except Exception as e:
            logger.error(f"pgvector similarity_search failed: {e}")
            return self._memory_fallback.similarity_search(graph_name, query_embedding, top_k)

    def delete_embeddings(self, graph_name: str, node_ids: List[str]) -> None:
        if not self._connected:
            self._memory_fallback.delete_embeddings(graph_name, node_ids)
            return

        if not node_ids:
            return

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM retell_embeddings WHERE graph_name = %s AND node_id = ANY(%s);",
                    (graph_name, node_ids)
                )
        except Exception as e:
            logger.error(f"pgvector delete_embeddings failed: {e}")

    def get_status(self, graph_name: str) -> Dict[str, Any]:
        if not self._connected:
            status = self._memory_fallback.get_status(graph_name)
            status["backend"] = "memory_fallback (disconnected from pgvector)"
            return status

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM retell_embeddings WHERE graph_name = %s;",
                    (graph_name,)
                )
                count = cur.fetchone()[0]
                return {
                    "backend": "pgvector",
                    "graph_name": graph_name,
                    "total_vectors": count
                }
        except Exception as e:
            logger.error(f"pgvector get_status failed: {e}")
            return {
                "backend": "pgvector (error)",
                "graph_name": graph_name,
                "error": str(e)
            }
