"""
Tests for FastAPI hosted API endpoints.
"""

from fastapi.testclient import TestClient
from retelldb.server.app import create_app
from retelldb.config import RetellDBConfig, StorageConfig


def test_api_endpoints():
    test_config = RetellDBConfig(
        storage=StorageConfig(backend="memory"),
        allowed_origins=["*"]
    )
    app = create_app(test_config)
    client = TestClient(app)

    # 1. Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "embedding_mode" in resp.json()

    # 2. Ingest document
    ingest_payload = {
        "content": "# Quickstart Guide\n\n## Getting Started\n\n```python\nprint('welcome')\n```",
        "format": "markdown",
        "audience": "developer",
        "title": "Quickstart Guide"
    }
    resp = client.post("/v1/ingest", json=ingest_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["nodes_created"] > 0

    # 3. Status check
    resp = client.get("/v1/status")
    assert resp.status_code == 200
    stat = resp.json()
    assert stat["dev_graph"]["total_nodes"] > 0
    assert "embedding_subsystem" in stat

    # 4. Query document
    query_payload = {
        "query": "quickstart getting started guide",
        "audience": "developer",
        "max_hops": 2,
        "top_k": 3,
        "include_edges": True
    }
    resp = client.post("/v1/query", json=query_payload)
    assert resp.status_code == 200
    q_data = resp.json()
    assert q_data["success"] is True
    assert len(q_data["result"]["nodes"]) > 0


def test_api_authentication():
    secure_config = RetellDBConfig(
        storage=StorageConfig(backend="memory"),
        api_key="secret-token-123"
    )
    app = create_app(secure_config)
    client = TestClient(app)

    # Unauthorized request
    resp = client.get("/v1/status")
    assert resp.status_code == 401

    # Authorized request
    resp = client.get("/v1/status", headers={"X-API-Key": "secret-token-123"})
    assert resp.status_code == 200
