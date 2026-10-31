"""
Integration tests for RetellDB — require live Docker Compose backends.

Run with:
    docker compose up -d
    pytest tests/test_integration.py -m integration -v

Unit suite remains hermetic; no Docker required for it.
"""

import os
import logging
import pytest

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://postgres:postgres@localhost:5432/retelldb",
)

TEST_GRAPH_DEV = "integ_dev_graph"
TEST_GRAPH_USER = "integ_user_graph"


# ---------------------------------------------------------------------------
# Pytest marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require live Docker Compose backends",
    )


# ---------------------------------------------------------------------------
# Helper: build a StorageConfig pointed at the compose services
# ---------------------------------------------------------------------------


def _make_storage_config(**overrides):
    from retelldb.config import StorageConfig

    return StorageConfig(
        backend="falkordb",
        falkordb_host=FALKORDB_HOST,
        falkordb_port=FALKORDB_PORT,
        falkordb_password=None,
        pg_dsn=PG_DSN,
        embedding_dimension=4,  # tiny dim for tests — avoids real embedding calls
        dev_graph_name=TEST_GRAPH_DEV,
        user_graph_name=TEST_GRAPH_USER,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Session-scoped live-backend fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def falkor_store():
    """Live FalkorDB store connected to compose container."""
    from retelldb.storage.falkordb_store import FalkorDBStore

    cfg = _make_storage_config()
    store = FalkorDBStore(config=cfg)
    if not store._connected:
        pytest.skip("FalkorDB not reachable — is `docker compose up` running?")
    yield store
    # Teardown: wipe test graphs
    for gname in (TEST_GRAPH_DEV, TEST_GRAPH_USER):
        try:
            g = store._get_graph(gname)
            g.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass


@pytest.fixture(scope="session")
def pg_store():
    """Live pgvector store connected to compose container."""
    from retelldb.storage.pgvector_store import PGVectorStore

    cfg = _make_storage_config()
    store = PGVectorStore(config=cfg)
    if not store._connected:
        pytest.skip("pgvector not reachable — is `docker compose up` running?")
    yield store
    # Teardown: wipe test rows
    try:
        with store._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM retell_embeddings WHERE graph_name IN (%s, %s);",
                (TEST_GRAPH_DEV, TEST_GRAPH_USER),
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Re-usable node/edge builders (function-scoped fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_concept_node():
    from retelldb.models.schema import GraphNode, NodeType, AudienceType

    return GraphNode(
        id="integ-node-concept-001",
        title="Authentication",
        type=NodeType.CONCEPT,
        content="Core authentication concept for RetellDB.",
        audience=AudienceType.DEVELOPER,
        tags=["auth", "security"],
        source_uri="integ://test/auth",
        locator="section:auth",
    )


@pytest.fixture()
def sample_tag_node():
    from retelldb.models.schema import GraphNode, NodeType, AudienceType

    return GraphNode(
        id="tag:developer:auth",
        title="auth",
        type=NodeType.TAG,
        content="Semantic tag node: auth",
        audience=AudienceType.DEVELOPER,
        tags=["auth"],
    )


@pytest.fixture()
def sample_endpoint_node():
    from retelldb.models.schema import GraphNode, NodeType, AudienceType

    return GraphNode(
        id="integ-node-endpoint-001",
        title="POST /v1/auth/refresh",
        type=NodeType.API_ENDPOINT,
        content="Refresh access token endpoint.",
        audience=AudienceType.DEVELOPER,
        tags=["auth", "tokens"],
        source_uri="integ://test/auth",
        locator="endpoint:/v1/auth/refresh",
    )


@pytest.fixture()
def sample_edge(sample_concept_node, sample_endpoint_node):
    from retelldb.models.schema import GraphEdge, EdgeType

    return GraphEdge(
        id="integ-edge-001",
        source_id=sample_endpoint_node.id,
        target_id=sample_concept_node.id,
        type=EdgeType.DEPENDS_ON,
        weight=1.0,
    )


@pytest.fixture()
def sample_tagged_as_edge(sample_concept_node, sample_tag_node):
    from retelldb.models.schema import GraphEdge, EdgeType

    return GraphEdge(
        id="integ-tagged-001",
        source_id=sample_concept_node.id,
        target_id=sample_tag_node.id,
        type=EdgeType.TAGGED_AS,
        weight=1.0,
    )


# ===========================================================================
# FALKORDB TESTS
# ===========================================================================


@pytest.mark.integration
class TestFalkorDBAddNode:
    """FalkorDB add_node / get_node round-trip."""

    def test_add_and_get_node(self, falkor_store, sample_concept_node):
        falkor_store.add_node(TEST_GRAPH_DEV, sample_concept_node)
        fetched = falkor_store.get_node(TEST_GRAPH_DEV, sample_concept_node.id)

        assert fetched is not None, "get_node returned None after add_node"
        assert fetched.id == sample_concept_node.id
        assert fetched.title == sample_concept_node.title
        assert fetched.type == sample_concept_node.type

    def test_get_nonexistent_node_returns_none(self, falkor_store):
        result = falkor_store.get_node(TEST_GRAPH_DEV, "no-such-node-xyz")
        assert result is None


@pytest.mark.integration
class TestFalkorDBAddEdge:
    """FalkorDB add_edge and get_edges round-trip."""

    def test_add_edge_and_get_edges(
        self,
        falkor_store,
        sample_concept_node,
        sample_endpoint_node,
        sample_edge,
    ):
        falkor_store.add_node(TEST_GRAPH_DEV, sample_concept_node)
        falkor_store.add_node(TEST_GRAPH_DEV, sample_endpoint_node)
        falkor_store.add_edge(TEST_GRAPH_DEV, sample_edge)

        edges = falkor_store.get_edges(TEST_GRAPH_DEV, sample_endpoint_node.id)
        edge_ids = [e.id for e in edges]

        assert sample_edge.id in edge_ids, (
            f"Expected edge {sample_edge.id!r} in {edge_ids!r}"
        )


@pytest.mark.integration
class TestFalkorDBTagTraversal:
    """FalkorDB tag traversal produces expected neighbours."""

    def test_tag_traversal_finds_connected_node(
        self,
        falkor_store,
        sample_concept_node,
        sample_tag_node,
        sample_tagged_as_edge,
    ):
        falkor_store.add_node(TEST_GRAPH_DEV, sample_concept_node)
        falkor_store.add_node(TEST_GRAPH_DEV, sample_tag_node)
        falkor_store.add_edge(TEST_GRAPH_DEV, sample_tagged_as_edge)

        node_hops, edges = falkor_store.tag_traversal(
            TEST_GRAPH_DEV, ["auth"], max_hops=2
        )

        retrieved_ids = {n.id for n, _ in node_hops}
        assert sample_concept_node.id in retrieved_ids, (
            f"tag_traversal should reach concept node; got ids: {retrieved_ids}"
        )

    def test_tag_traversal_unknown_tag_returns_empty(self, falkor_store):
        node_hops, edges = falkor_store.tag_traversal(
            TEST_GRAPH_DEV, ["completely_unknown_tag_xyz"], max_hops=2
        )
        assert node_hops == []
        assert edges == []


@pytest.mark.integration
class TestFalkorDBSubgraph:
    """FalkorDB get_subgraph_between returns internal edges."""

    def test_subgraph_between(
        self,
        falkor_store,
        sample_concept_node,
        sample_endpoint_node,
        sample_edge,
    ):
        falkor_store.add_node(TEST_GRAPH_DEV, sample_concept_node)
        falkor_store.add_node(TEST_GRAPH_DEV, sample_endpoint_node)
        falkor_store.add_edge(TEST_GRAPH_DEV, sample_edge)

        edges = falkor_store.get_subgraph_between(
            TEST_GRAPH_DEV,
            [sample_concept_node.id, sample_endpoint_node.id],
        )

        assert len(edges) >= 1, "subgraph_between should return at least one edge"
        edge_ids = [e.id for e in edges]
        assert sample_edge.id in edge_ids

    def test_subgraph_empty_node_ids_returns_empty(self, falkor_store):
        assert falkor_store.get_subgraph_between(TEST_GRAPH_DEV, []) == []


@pytest.mark.integration
class TestFalkorDBStatus:
    """FalkorDB get_status reports correct backend name and has node counts."""

    def test_status_reports_falkordb_backend(self, falkor_store):
        status = falkor_store.get_status(TEST_GRAPH_DEV)
        assert status["backend"] == "falkordb"
        assert "total_nodes" in status
        assert "total_edges" in status
        assert isinstance(status["total_nodes"], int)

    def test_status_counts_increase_after_writes(self, falkor_store):
        from retelldb.models.schema import GraphNode, NodeType, AudienceType

        new_node = GraphNode(
            id="integ-status-count-check",
            title="Status Check Node",
            type=NodeType.GUIDE,
            content="used to verify count increment",
            audience=AudienceType.DEVELOPER,
        )
        falkor_store.add_node(TEST_GRAPH_DEV, new_node)
        status = falkor_store.get_status(TEST_GRAPH_DEV)
        assert status["total_nodes"] >= 1


@pytest.mark.integration
class TestFalkorDBIdempotentReIngestion:
    """Re-ingesting the same source URI must not duplicate nodes."""

    def test_idempotent_reingest_same_source(self, falkor_store):
        from retelldb.models.schema import GraphNode, NodeType, AudienceType

        src = "integ://idempotency-test"
        node = GraphNode(
            id="integ-idempotent-001",
            title="Idempotency Doc",
            type=NodeType.CONCEPT,
            content="Same content, ingested twice.",
            audience=AudienceType.DEVELOPER,
            source_uri=src,
            locator="section:idempotency",
        )

        falkor_store.add_node(TEST_GRAPH_DEV, node)
        # MERGE semantics: second write must not create duplicate
        falkor_store.add_node(TEST_GRAPH_DEV, node)

        nodes = falkor_store.get_nodes_by_source(TEST_GRAPH_DEV, src)
        ids = [n.id for n in nodes]
        assert ids.count("integ-idempotent-001") == 1, (
            f"Expected exactly one node with this id after double-write; got {ids}"
        )


# ===========================================================================
# PGVECTOR TESTS
# ===========================================================================


@pytest.mark.integration
class TestPGVectorAdd:
    """pgvector add_embedding / similarity_search round-trip."""

    def test_add_and_search(self, pg_store):
        embedding = [0.1, 0.9, 0.3, 0.4]
        pg_store.add_embedding(
            TEST_GRAPH_DEV,
            "pg-node-001",
            embedding,
            {"title": "Auth concept"},
        )

        results = pg_store.similarity_search(TEST_GRAPH_DEV, embedding, top_k=5)
        node_ids = [r[0] for r in results]
        assert "pg-node-001" in node_ids, (
            f"Newly added embedding not found; got {node_ids}"
        )

    def test_scores_are_non_negative(self, pg_store):
        embedding = [0.5, 0.5, 0.5, 0.5]
        pg_store.add_embedding(TEST_GRAPH_DEV, "pg-node-score-check", embedding)
        results = pg_store.similarity_search(TEST_GRAPH_DEV, embedding, top_k=10)
        for _, score in results:
            assert score >= 0.0, f"Negative score encountered: {score}"

    def test_search_empty_graph_returns_empty(self, pg_store):
        results = pg_store.similarity_search(
            "integ_empty_graph_xyz", [0.1, 0.2, 0.3, 0.4], top_k=5
        )
        assert results == []


@pytest.mark.integration
class TestPGVectorDelete:
    """pgvector delete_embeddings removes records."""

    def test_delete_removes_embedding(self, pg_store):
        embedding = [0.2, 0.3, 0.5, 0.1]
        pg_store.add_embedding(TEST_GRAPH_DEV, "pg-node-delete-me", embedding)

        pg_store.delete_embeddings(TEST_GRAPH_DEV, ["pg-node-delete-me"])

        results = pg_store.similarity_search(TEST_GRAPH_DEV, embedding, top_k=50)
        ids = [r[0] for r in results]
        assert "pg-node-delete-me" not in ids, "Deleted embedding still in search results"

    def test_delete_empty_list_is_noop(self, pg_store):
        """delete_embeddings([]) must not raise."""
        pg_store.delete_embeddings(TEST_GRAPH_DEV, [])


@pytest.mark.integration
class TestPGVectorStatus:
    """pgvector get_status reports correct backend."""

    def test_status_reports_pgvector_backend(self, pg_store):
        status = pg_store.get_status(TEST_GRAPH_DEV)
        assert status["backend"] == "pgvector"
        assert "total_vectors" in status
        assert isinstance(status["total_vectors"], int)


# ===========================================================================
# AUDIENCE ISOLATION ACROSS REAL BACKENDS
# ===========================================================================


@pytest.mark.integration
class TestAudienceIsolation:
    """Nodes written to dev_graph must not appear in user_graph and vice-versa."""

    def test_dev_node_not_in_user_graph(self, falkor_store):
        from retelldb.models.schema import GraphNode, NodeType, AudienceType

        dev_node = GraphNode(
            id="isolation-dev-node-001",
            title="Developer Only Concept",
            type=NodeType.CONCEPT,
            content="Only for devs.",
            audience=AudienceType.DEVELOPER,
            source_uri="integ://isolation-test",
        )
        falkor_store.add_node(TEST_GRAPH_DEV, dev_node)

        user_result = falkor_store.get_node(TEST_GRAPH_USER, dev_node.id)
        assert user_result is None, (
            "Dev node leaked into user_graph — audience isolation broken"
        )

    def test_user_node_not_in_dev_graph(self, falkor_store):
        from retelldb.models.schema import GraphNode, NodeType, AudienceType

        user_node = GraphNode(
            id="isolation-user-node-001",
            title="User Only Guide",
            type=NodeType.GUIDE,
            content="Only for users.",
            audience=AudienceType.USER,
            source_uri="integ://isolation-test-user",
        )
        falkor_store.add_node(TEST_GRAPH_USER, user_node)

        dev_result = falkor_store.get_node(TEST_GRAPH_DEV, user_node.id)
        assert dev_result is None, (
            "User node leaked into dev_graph — audience isolation broken"
        )

    def test_vector_namespace_isolation(self, pg_store):
        """Vectors stored under dev graph must not appear in user graph search."""
        emb = [0.8, 0.1, 0.0, 0.1]
        pg_store.add_embedding(TEST_GRAPH_DEV, "vec-dev-only-001", emb)

        user_results = pg_store.similarity_search(TEST_GRAPH_USER, emb, top_k=50)
        user_ids = [r[0] for r in user_results]
        assert "vec-dev-only-001" not in user_ids, (
            "Dev embedding visible in user graph namespace — isolation broken"
        )


# ===========================================================================
# FALLBACK BEHAVIOUR WHEN BACKEND IS UNREACHABLE
# ===========================================================================


@pytest.mark.integration
class TestMemoryFallback:
    """When backends are unreachable the stores must fall back to memory and log a warning."""

    def test_falkordb_fallback_on_bad_host(self, caplog):
        """FalkorDBStore with a bad host must fall back to MemoryGraphStore and emit a warning."""
        from retelldb.config import StorageConfig
        from retelldb.storage.falkordb_store import FalkorDBStore

        bad_cfg = StorageConfig(
            falkordb_host="127.0.0.1",
            falkordb_port=19999,  # nothing listening here
            pg_dsn=PG_DSN,
        )
        with caplog.at_level(logging.WARNING, logger="retelldb.falkordb"):
            store = FalkorDBStore(config=bad_cfg)

        assert not store._connected, "Store should not be connected to a bad host"
        assert store._memory_fallback is not None, "Memory fallback must be set"

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(
            "fallback" in m.lower() or "unavailable" in m.lower() or "in-memory" in m.lower()
            for m in warning_messages
        ), f"Expected fallback warning; got: {warning_messages}"

    def test_falkordb_fallback_serves_add_and_get(self, caplog):
        """Ops on a disconnected FalkorDBStore are transparently served by MemoryGraphStore."""
        from retelldb.config import StorageConfig
        from retelldb.storage.falkordb_store import FalkorDBStore
        from retelldb.models.schema import GraphNode, NodeType, AudienceType

        bad_cfg = StorageConfig(
            falkordb_host="127.0.0.1",
            falkordb_port=19999,
            pg_dsn=PG_DSN,
        )
        store = FalkorDBStore(config=bad_cfg)
        assert not store._connected

        node = GraphNode(
            id="fallback-node-001",
            title="Fallback Node",
            type=NodeType.CONCEPT,
            content="Lives in memory fallback.",
            audience=AudienceType.DEVELOPER,
        )
        store.add_node(TEST_GRAPH_DEV, node)
        fetched = store.get_node(TEST_GRAPH_DEV, node.id)
        assert fetched is not None, "Memory fallback should serve get_node"
        assert fetched.id == node.id

    def test_pgvector_fallback_on_bad_dsn(self, caplog):
        """PGVectorStore with a bad DSN must fall back to MemoryVectorStore and emit a warning."""
        from retelldb.config import StorageConfig
        from retelldb.storage.pgvector_store import PGVectorStore

        bad_cfg = StorageConfig(
            falkordb_host=FALKORDB_HOST,
            falkordb_port=FALKORDB_PORT,
            pg_dsn="postgresql://postgres:postgres@127.0.0.1:19998/retelldb",
            embedding_dimension=4,
        )
        with caplog.at_level(logging.WARNING, logger="retelldb.pgvector"):
            store = PGVectorStore(config=bad_cfg)

        assert not store._connected, "Store should not be connected with a bad DSN"
        assert store._memory_fallback is not None, "Memory fallback must be set"

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(
            "fallback" in m.lower() or "connect" in m.lower()
            for m in warning_messages
        ), f"Expected fallback warning; got: {warning_messages}"

    def test_pgvector_fallback_serves_search(self):
        """Fallback PGVectorStore must serve similarity_search via in-memory store."""
        from retelldb.config import StorageConfig
        from retelldb.storage.pgvector_store import PGVectorStore

        bad_cfg = StorageConfig(
            pg_dsn="postgresql://postgres:postgres@127.0.0.1:19998/retelldb",
            embedding_dimension=4,
        )
        store = PGVectorStore(config=bad_cfg)
        assert not store._connected

        emb = [0.3, 0.7, 0.1, 0.5]
        store.add_embedding(TEST_GRAPH_DEV, "fallback-vec-001", emb)
        results = store.similarity_search(TEST_GRAPH_DEV, emb, top_k=5)
        ids = [r[0] for r in results]
        assert "fallback-vec-001" in ids, "Memory fallback must serve similarity_search"


# ===========================================================================
# REGRESSION: fresh database initialisation order
# ===========================================================================


@pytest.mark.integration
class TestFreshDatabaseInit:
    """
    A brand new database has no vector type until the extension is created, so the store
    must run its schema first and register the client adapter second. Getting the order
    backwards made every pgvector call fall back to memory against a healthy Postgres.
    """

    def test_store_connects_on_a_database_without_the_extension(self):
        psycopg = pytest.importorskip("psycopg")
        from retelldb.storage.pgvector_store import PGVectorStore

        base, _ = PG_DSN.rsplit("/", 1)
        try:
            admin = psycopg.connect(f"{base}/postgres", autocommit=True, connect_timeout=3)
        except Exception:
            pytest.skip("postgres not reachable — is `docker compose up` running?")

        fresh_db = "retelldb_init_order_test"
        with admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{fresh_db}" WITH (FORCE)')
            admin.execute(f'CREATE DATABASE "{fresh_db}"')

        try:
            store = PGVectorStore(
                config=_make_storage_config(pg_dsn=f"{base}/{fresh_db}")
            )
            assert store._connected is True, (
                "a database without the vector extension yet must not fall back to memory"
            )

            store.add_embedding(TEST_GRAPH_DEV, "fresh-init-001", [0.2, 0.2, 0.2, 0.2])
            hits = store.similarity_search(TEST_GRAPH_DEV, [0.2, 0.2, 0.2, 0.2], top_k=1)
            assert [node_id for node_id, _score in hits] == ["fresh-init-001"]
        finally:
            with admin:
                admin.execute(f'DROP DATABASE IF EXISTS "{fresh_db}" WITH (FORCE)')
