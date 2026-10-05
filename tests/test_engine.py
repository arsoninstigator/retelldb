"""
End-to-end tests for IngestionPipeline, RetrievalEngine, and RetellDB client.
"""

from retelldb.client import RetellDB
from retelldb.models.schema import AudienceType, NodeType
from retelldb.models.query import QueryStrategy


def test_end_to_end_retrieval():
    client = RetellDB(use_memory=True)

    # 1. Ingest developer documentation
    dev_doc = """---
title: Token Refresh Workflow
tags: [auth, tokens, refresh]
audience: developer
---

# Token Refresh Workflow

Concepts behind session tokens and how to refresh them securely.

## POST /v1/auth/refresh

API endpoint to exchange expired access token for a new one.

```python
from retelldb import AuthClient
client = AuthClient(refresh_token="xyz")
new_token = client.refresh()
```

## Error 401 Invalid Token

Raised when refresh token is malformed or revoked.
"""

    resp = client.ingest(
        content=dev_doc,
        format="markdown",
        audience="developer",
        title="Token Refresh Workflow",
        source_uri="docs/auth.md"
    )

    assert resp.success is True
    assert resp.nodes_created >= 4
    assert resp.tags_created >= 3

    # Check status
    status = client.status()
    assert status.dev_graph["total_nodes"] > 0
    assert status.embedding_subsystem is not None

    # 2. Test Tag Traversal (precise / tagged query)
    tag_result = client.query(
        query="how do i refresh auth tokens",
        audience="developer",
        max_hops=3,
        top_k=5,
        include_edges=True
    )

    assert tag_result.retrieval_path in (QueryStrategy.TAG_TRAVERSAL, QueryStrategy.HYBRID)
    assert len(tag_result.nodes) > 0
    node_titles = [sn.node.title for sn in tag_result.nodes]
    assert any("POST /v1/auth/refresh" in t or "Token Refresh" in t for t in node_titles)
    assert len(tag_result.edges) > 0

    # 3. Test Vector Fallback (vague / latent query without exact tag match)
    vague_result = client.query(
        query="my credentials expired while calling the service",
        audience="developer",
        top_k=3
    )

    assert vague_result.retrieval_path == QueryStrategy.VECTOR_FALLBACK
    assert len(vague_result.nodes) > 0

    # 4. Audience Isolation Test (user query should route to user_graph which is empty)
    user_result = client.query(
        query="how to update credit card on dashboard settings",
        audience="user"
    )
    assert user_result.target_graph == "user_graph"
    assert len(user_result.nodes) == 0


def test_idempotent_reingestion_and_upsert():
    """Verify re-ingesting the same document produces deterministic IDs without duplicate nodes."""
    client = RetellDB(use_memory=True)

    doc_v1 = """---
title: Rate Limits
tags: [rate-limits, throttling]
---
# Rate Limits
## Standard Limits
100 requests per minute.
## Burst Limits
200 requests for 10 seconds.
"""

    # Ingest v1
    resp1 = client.ingest(
        content=doc_v1,
        format="markdown",
        audience="developer",
        source_uri="docs/rate-limits.md"
    )
    stat1 = client.status()
    initial_nodes = stat1.dev_graph["total_nodes"]
    initial_edges = stat1.dev_graph["total_edges"]

    # Re-ingest the exact same doc (should be completely idempotent)
    resp2 = client.ingest(
        content=doc_v1,
        format="markdown",
        audience="developer",
        source_uri="docs/rate-limits.md"
    )
    stat2 = client.status()
    assert stat2.dev_graph["total_nodes"] == initial_nodes
    assert stat2.dev_graph["total_edges"] == initial_edges

    # Ingest v2 where "Burst Limits" section was removed (upsert reconciliation)
    doc_v2 = """---
title: Rate Limits
tags: [rate-limits, throttling]
---
# Rate Limits
## Standard Limits
100 requests per minute.
"""
    resp3 = client.ingest(
        content=doc_v2,
        format="markdown",
        audience="developer",
        source_uri="docs/rate-limits.md"
    )
    stat3 = client.status()
    # The Burst Limits node should be reconciled and removed
    assert stat3.dev_graph["total_nodes"] < initial_nodes


def test_audience_isolation_frontmatter_override():
    """Verify that frontmatter audience overrides route strictly to the target graph."""
    client = RetellDB(use_memory=True)

    user_doc = """---
title: Billing and Invoicing
tags: [billing, invoices]
audience: user
---
# Billing and Invoicing
How to download PDF receipts for your account.
"""

    # Ingest call mistakenly specifies audience="developer", but frontmatter declares audience: user
    resp = client.ingest(
        content=user_doc,
        format="markdown",
        audience="developer",  # caller argument
        source_uri="docs/billing.md"
    )

    stat = client.status()
    # dev_graph must be completely empty!
    assert stat.dev_graph["total_nodes"] == 0
    # user_graph must contain the nodes!
    assert stat.user_graph["total_nodes"] > 0


def test_hop_distance_decay_scoring():
    """Verify that graph traversal scores decay with hop distance from the matched tag."""
    client = RetellDB(use_memory=True)

    doc = """---
title: Payment Webhooks
tags: [payments]
---
# Payment Webhooks
Webhook concepts for payments.

## Handling Events
Process incoming webhook payloads.

```python
# Code example
def handle_event(payload): pass
```
"""
    client.ingest(content=doc, format="markdown", audience="developer")

    result = client.query(
        query="payments",
        audience="developer",
        max_hops=3,
        top_k=5
    )

    assert result.retrieval_path in (QueryStrategy.TAG_TRAVERSAL, QueryStrategy.HYBRID)
    # Check that scores decrease or remain consistent with distance
    scores = [sn.score for sn in result.nodes]
    assert scores[0] >= scores[-1]
    assert all(s <= 1.0 for s in scores)
