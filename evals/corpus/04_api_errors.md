---
title: Common API Errors
tags: [errors, api, debugging, http, 401, unauthorized, rate-limit]
audience: developer
---

# Common API Errors

This document covers the most frequent error responses from the RetellDB REST API and how to resolve them.

## 401 Unauthorized

Returned when the API key is missing or revoked.

```json
{
  "status": 401,
  "detail": "API key is invalid or has been revoked."
}
```

**Fix**: Verify the key is active in the dashboard and that the `X-API-Key` header is present.

## 404 Not Found

Returned when the requested graph name or node ID does not exist.

```json
{
  "status": 404,
  "detail": "Graph 'my_graph' not found."
}
```

**Fix**: Check that the graph was created by ingesting at least one document. Graph instances are created lazily on first write.

## 422 Unprocessable Entity

Returned when the request payload fails schema validation.

```json
{
  "status": 422,
  "detail": [{"loc": ["body", "format"], "msg": "value is not a valid enum member"}]
}
```

**Fix**: The `format` field must be one of: `markdown`, `mdx`, `openapi`, `asyncapi`, `html`, `notion`.

## 429 Rate Limit Exceeded

The API enforces per-key rate limits. Retry with exponential back-off:

```python
import time
import retelldb

client = retelldb.RetellDB()
for attempt in range(5):
    try:
        result = client.query("token refresh flow")
        break
    except retelldb.RetellDBError as e:
        if e.status_code == 429:
            time.sleep(2 ** attempt)
        else:
            raise
```

## 500 Internal Server Error

An unexpected error occurred on the server. Always include the `X-Request-ID` header value from the response when filing a support ticket.
