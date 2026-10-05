"""
Tests for document normalizers.
"""

from retelldb.ingestion.normalizers.markdown import MarkdownNormalizer
from retelldb.ingestion.normalizers.openapi import OpenAPINormalizer
from retelldb.ingestion.normalizers.html import HTMLNormalizer
from retelldb.ingestion.normalizers.notion import NotionNormalizer
from retelldb.models.schema import NodeType, EdgeType, AudienceType


def test_markdown_normalizer():
    content = """---
title: Authentication Guide
tags: [auth, tokens, security]
audience: developer
---

# Authentication Guide

Learn how to authenticate with the API.

## POST /v1/auth/token

Create a session token.

```python
import retelldb
token = retelldb.auth("key")
```

## Error Troubleshooting

Common authentication errors like 401 Unauthorized.
"""
    normalizer = MarkdownNormalizer()
    doc = normalizer.normalize(content, audience=AudienceType.DEVELOPER)

    assert doc.document_title == "Authentication Guide"
    assert "auth" in doc.global_tags
    assert len(doc.nodes) >= 4  # Root, POST /v1/auth/token, Example, Error

    types = {n.type for n in doc.nodes}
    assert NodeType.API_ENDPOINT in types
    assert NodeType.EXAMPLE in types
    assert NodeType.ERROR in types

    # Check that example connects to parent with EXAMPLE_OF
    example_edges = [e for e in doc.edges if e.type == EdgeType.EXAMPLE_OF]
    assert len(example_edges) >= 1


def test_openapi_normalizer():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Pet Store API", "version": "1.0.0"},
        "paths": {
            "/pets": {
                "get": {
                    "summary": "List all pets",
                    "tags": ["pets"],
                    "responses": {
                        "200": {"description": "List of pets"},
                        "500": {"description": "Server error"}
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "Pet": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "name": {"type": "string"}}
                }
            }
        }
    }
    import json
    normalizer = OpenAPINormalizer()
    doc = normalizer.normalize(json.dumps(spec), audience=AudienceType.DEVELOPER)

    assert doc.document_title == "Pet Store API (v1.0.0)"
    node_types = {n.type for n in doc.nodes}
    assert NodeType.API_ENDPOINT in node_types
    assert NodeType.CONCEPT in node_types
    assert NodeType.ERROR in node_types


def test_html_normalizer():
    html_content = """
    <html>
        <head><title>Getting Started Guide</title></head>
        <body>
            <nav><a href="/quickstart">Quickstart</a></nav>
            <h2>Installation</h2>
            <p>Install via pip or npm.</p>
            <pre><code class="language-bash">pip install retelldb</code></pre>
        </body>
    </html>
    """
    normalizer = HTMLNormalizer()
    doc = normalizer.normalize(html_content, audience=AudienceType.DEVELOPER)

    assert doc.document_title == "Getting Started Guide"
    assert any(n.type == NodeType.EXAMPLE for n in doc.nodes)


def test_notion_normalizer():
    notion_json = {
        "results": [
            {
                "type": "heading_1",
                "heading_1": {"rich_text": [{"plain_text": "Notion Project Setup"}]}
            },
            {
                "type": "code",
                "code": {"language": "python", "rich_text": [{"plain_text": "print('hello')"}]}
            }
        ]
    }
    import json
    normalizer = NotionNormalizer()
    doc = normalizer.normalize(json.dumps(notion_json), audience=AudienceType.DEVELOPER)
    assert any(n.title == "Notion Project Setup" for n in doc.nodes)
    assert any(n.type == NodeType.EXAMPLE for n in doc.nodes)
