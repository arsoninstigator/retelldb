---
title: Token Refresh Workflow
tags: [auth, tokens, refresh]
audience: developer
---

# Token Refresh Workflow

When an access token expires after 60 minutes, the client uses a refresh token to generate a new key pair without prompting user re-login.

## POST /v1/auth/refresh

Endpoint to exchange a refresh token for a fresh access token.

```python
import retelldb
client = retelldb.RetellDB()
new_token = client.auth.refresh(refresh_token="ref_abc123")
```

## Error 401 Revoked Token

Returned when the refresh token has expired or was revoked by an administrator.
