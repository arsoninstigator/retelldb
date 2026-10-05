---
title: Authentication Architecture
tags: [auth, security, tokens]
audience: developer
---

# Authentication Architecture

RetellDB authentication relies on asymmetric JWT session tokens and API keys.

## API Key Authentication

API keys authenticate machine-to-machine requests. Include the key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: retell_live_key_9988" https://api.retelldb.io/v1/status
```

## Session Tokens

For interactive sessions, client applications obtain short-lived JWT tokens valid for 1 hour.
