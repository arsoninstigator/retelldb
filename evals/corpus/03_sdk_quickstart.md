---
title: SDK Quickstart
tags: [sdk, quickstart, python, setup]
audience: developer
---

# SDK Quickstart

RetellDB ships a Python SDK that wraps all REST endpoints with typed request/response models.

## Installation

```bash
pip install retelldb
```

## Initializing the Client

```python
import retelldb

client = retelldb.RetellDB(api_key="retell_live_key_9988")
```

The client reads `RETELLDB_API_KEY` from the environment automatically if no explicit key is passed.

## Ingesting a Document

```python
response = client.ingest(
    content=open("my_docs/auth.md").read(),
    format="markdown",
    audience="developer",
    title="Authentication Guide",
    source_uri="file://auth.md",
)
print(response.nodes_created)  # e.g. 4
```

## Querying

```python
result = client.query("how do I refresh an access token?", audience="developer")
for node in result.nodes:
    print(node.title, node.score)
```

## Error Handling

All SDK methods raise `retelldb.RetellDBError` on HTTP 4xx/5xx. The exception carries `.status_code` and `.detail`.

```python
try:
    client.query("what is the billing cycle?", audience="developer")
except retelldb.RetellDBError as e:
    print(f"Error {e.status_code}: {e.detail}")
```
