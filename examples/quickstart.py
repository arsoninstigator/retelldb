"""
RetellDB Quickstart Example.
Demonstrates ingesting documentation and querying via the graph routing layer.
"""

import json
from retelldb import RetellDB

def main():
    print("=" * 60)
    print("RetellDB Quickstart Demo")
    print("=" * 60)

    # Initialize RetellDB in-memory client
    client = RetellDB(use_memory=True)

    # 1. Sample developer documentation markdown
    sample_doc = """---
title: Authentication and Session Tokens
tags: [auth, tokens, security, session]
audience: developer
---

# Authentication and Session Tokens

Learn how to authenticate with the Retell API using bearer session tokens.

## Concept: Token Lifecycle

Tokens are short-lived credentials (valid for 60 minutes). When expired,
clients must use the refresh flow rather than requesting a new login.

## POST /v1/auth/refresh

Exchange an expired access token using your client refresh token.

### Python SDK Example

```python
import retelldb
client = retelldb.Client(api_key="your_api_key")
session = client.auth.refresh_token("refresh_tok_123")
print("New token:", session.access_token)
```

## Error 401 Unauthorized

Common failure cases when refreshing session tokens:
- Refresh token has been revoked
- Refresh token expired after 30 days of inactivity
"""

    print("\n1. Ingesting documentation into Developer Graph...")
    ingest_resp = client.ingest(
        content=sample_doc,
        format="markdown",
        audience="developer",
        title="Authentication and Session Tokens"
    )
    print(f"-> Ingestion completed in {ingest_resp.execution_time_ms}ms")
    print(f"   Nodes created: {ingest_resp.nodes_created}")
    print(f"   Edges created: {ingest_resp.edges_created}")
    print(f"   Tags created:  {ingest_resp.tags_created}")

    # Check status
    status = client.status()
    print("\nGraph Status:")
    print(f"   Dev Graph Nodes: {status.dev_graph['total_nodes']} (Tags: {status.dev_graph['total_tags']})")

    # 2. Precise / Tagged Query (triggers Semantic Tag Traversal)
    print("\n" + "-" * 60)
    print("2. Query 1 (Precise): 'how to refresh auth tokens in python sdk'")
    result1 = client.query(
        query="how to refresh auth tokens in python sdk",
        audience="developer",
        max_hops=3,
        top_k=4,
        include_edges=True
    )
    print(f"-> Target Graph:   {result1.target_graph}")
    print(f"-> Retrieval Path: {result1.retrieval_path.value}")
    print(f"-> Latency:        {result1.execution_time_ms}ms")
    print("-> Retrieved Nodes:")
    for idx, sn in enumerate(result1.nodes, 1):
        print(f"   [{idx}] ({sn.node.type.value}) {sn.node.title} - Score: {sn.score} (via {sn.retrieval_source})")

    if result1.edges:
        print("-> Connecting Edges:")
        for e in result1.edges[:5]:
            print(f"   {e.source_id[:12]}.. --[{e.type.value}]--> {e.target_id[:12]}..")

    # 3. Vague / Latent Query (triggers Associative Vector Fallback)
    print("\n" + "-" * 60)
    print("3. Query 2 (Vague/Latent): 'credentials got rejected during api call'")
    result2 = client.query(
        query="credentials got rejected during api call",
        audience="developer",
        top_k=3
    )
    print(f"-> Target Graph:   {result2.target_graph}")
    print(f"-> Retrieval Path: {result2.retrieval_path.value}")
    print(f"-> Latency:        {result2.execution_time_ms}ms")
    print("-> Retrieved Nodes:")
    for idx, sn in enumerate(result2.nodes, 1):
        print(f"   [{idx}] ({sn.node.type.value}) {sn.node.title} - Score: {sn.score} (via {sn.retrieval_source})")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
