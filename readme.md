# retelldb

> a graph-powered retrieval layer for agents that actually understand your docs.

---

## what is this

retelldb is a retrieval engine built specifically for agents that work with documentation.

not support tickets. not chat history. docs. developer references, api guides, sdk pages, user-facing tutorials, the whole thing.

the problem with most doc search is that it treats every page like an isolated blob of text. retelldb models your docs as a graph, so an agent can actually traverse relationships between concepts, understand that this endpoint replaced that one, follow a "depends on" edge from a tutorial to a reference page, and know whether it is talking to a developer or a regular user before it even starts retrieving.

it also does semantic tag matching and latent vector similarity, so it catches both the precise lookups and the vague ones.

the latent similarity side of this is spiritually close to what [memorium](https://github.com/tinkrland/memorium) is doing with associative memory -- building dense connections between ideas rather than just retrieving from a flat bucket. retelldb borrows that framing for the doc retrieval problem: when there is no clean tag match, the fallback is not keyword search, it is proximity through the associative space of your docs.

---

## the core idea

your docs have structure. retelldb keeps that structure instead of flattening it away.

- concepts link to other concepts
- api pages link to the sdk methods that implement them
- versioned nodes know what they replaced
- tags sit as first-class edges, not metadata strings
- developer docs and user docs live in separate graphs so they don't pollute each other

when an agent queries retelldb, it can traverse that structure rather than just scoring text similarity against a pile of chunks.

---

## what it handles

- markdown, mdx, openapi / asyncapi specs, notion, html docs
- two separate graphs per project: one for developer docs, one for user docs
- semantic tag traversal and vector similarity fallback, both in the same query
- a routing layer that decides which retrieval strategy to use per query (this is core, not something you wire up yourself)
- self-hostable open source library or hosted api, your choice

---

---

## the tag wrangler

tags are the part of retrieval that decides everything downstream, so retelldb treats
them as a typed, auditable ontology instead of string metadata.

**canonical tags, aliases, and provenance.** a raw tag is never rewritten. an observed
`dev` or `coding` stays exactly as it was on the node, and the resolution is written to
the graph as an `alias_of` edge from the alias tag node to its canonical tag node, weighted
by confidence. you can always answer "what did the human actually type, and what did we
decide it meant".

**ambiguity resolves to nothing.** if an alias is configured for two canonical tags, it
resolves to neither and is reported as ambiguous. the wrangler does not guess from context
at ingest time.

**perspective is a retrieval parameter.** nodes can be stamped with personas and a
sensitivity. a query carries `persona`, and scope is fail-closed: no declared persona means
persona-scoped content is not visible, private content is only visible to its own persona,
and everything withheld comes back in `blocked` with a reason.

**weighted traversal, not nearest neighbour.** every edge carries an explicit weight. a
walk multiplies them along the path, prunes edges below a floor, refuses `blocks` edges, and
cannot loop. the result includes the path that pulled each node in, with its weights.

**confidence tiers and structured doubt.** the engine returns a total confidence, the tier
it falls in (`high`, `uncertain`, `needs_clarification`, `unavailable`), the components that
dampened it, and the next action the agent should take. a weak match is never dressed up as
a confident answer.

**citations you can check.** every node carries source references (path, line range, content
type). the engine resolves them into citations, and verifies the cited span against the
source text when it is available.

### generating the ontology

hand writing aliases does not scale, so the taxonomy can be generated from the tags a corpus
actually produces, using a dataset and model provider (Adaption Labs by default):

```bash
export ADAPTIONLABS_API_KEY=...
retelldb taxonomy generate ./docs --out taxonomy.yaml --dry-run   # inspect the prompt first
retelldb taxonomy generate ./docs --out taxonomy.yaml
retelldb taxonomy validate taxonomy.yaml
export RETELLDB_TAXONOMY_PATH=taxonomy.yaml
```

Adaption Labs is the default provider. `--provider openai-compat` points the same
generator at any chat-completions endpoint (`--base-url`, `--model`), which is useful when
the primary provider is unavailable. `ADAPTIONLABS_BASE_URL` overrides the endpoint.

### the provider is a build dependency, never a runtime one

generation runs during a build and its output is a reviewed `taxonomy.yaml` that ships with
the release. nothing in the retrieval path calls a model provider, and the package layout
makes that structural rather than conventional:

- the provider lives in `retelldb/buildtime/`, which the runtime never imports. a test boots a
  fresh interpreter, imports `retelldb`, `retelldb.client` and `retelldb.wrangling`, and fails if
  anything under `retelldb.buildtime` loaded.
- `retelldb/wrangling/generator.py` is provider agnostic: it takes any object with
  `available()` and `complete(system, prompt)`, so the generator can be tested with a stub and
  the ontology logic carries no vendor.
- `assert_build_time()` runs on construction and on every completion. if
  `RETELLDB_DEPLOYMENT`, `RETELLDB_ENV`, `DEPLOYMENT_ENV` or `ENVIRONMENT` says `production`,
  `prod` or `live`, the transport raises instead of making the call. there is no override flag
  on purpose.

generation is honest by construction: every observed raw tag must come back classified, tags
left uncovered are reported, a failed model call fails the command instead of writing a
partial ontology, and nothing is aliased until you point the config at the reviewed file.

## quickstart

### 1. Installation

```bash
pip install -e .
```

For external database drivers (FalkorDB, pgvector, LiteLLM):
```bash
pip install -e ".[all]"
```

### 2. Python SDK

```python
from retelldb import RetellDB

# Initialize client (defaults to in-memory mode if no external DB is configured)
client = RetellDB()

# Ingest documentation
client.ingest(
    content="""---
title: Authentication
tags: [auth, tokens]
---
# Authentication
## POST /v1/auth/token
Create an API session token.
```python
token = client.auth()
```
""",
    format="markdown",
    audience="developer"
)

# Precise query -> routes to Semantic Tag Traversal
result = client.query("how to generate auth tokens in python")
print(result.retrieval_path)  # tag_traversal
for item in result.nodes:
    print(f"[{item.node.type}] {item.node.title} (score: {item.score})")

# Vague query -> routes to Latent Associative Vector Fallback
vague_result = client.query("my credentials failed to connect")
print(vague_result.retrieval_path)  # vector_fallback

# Perspective, confidence, and citations
scoped = client.query("what should i work on", persona="student")
print(scoped.confidence.tier)        # high | uncertain | needs_clarification | unavailable
print(scoped.confidence.next_actions)  # what to do when the tier is not "high"
print(scoped.blocked)                # what this perspective could not see, and why
for citation in scoped.citations:
    print(citation.location(), citation.verified)  # docs/auth.md:6-7 True/False/None
```

### 3. Hosted API & Server

Start the hosted API server:

```bash
retelldb serve --port 8000
```

Endpoints:
- `POST /v1/query` — Run query through routing layer
- `POST /v1/ingest` — Ingest Markdown, OpenAPI, Notion, or HTML
- `GET /v1/status` — Inspect dev and user graph sizes and node types
- `GET /health` — Health check

Query via HTTP or connect a remote SDK client:

```python
client = RetellDB(base_url="http://localhost:8000")
result = client.query("how to authenticate")
```

### 4. CLI Usage

```bash
# Ingest a documentation file
retelldb ingest ./docs/api.md --format markdown --audience developer

# Query documentation
retelldb query "how do i refresh a token" --max-hops 3 --top-k 5

# Check graph status
retelldb status
```

---

## architecture

```mermaid
flowchart TD
    subgraph Ingestion
        A[Raw Docs: MD / OpenAPI / Notion / HTML] --> B[Normalizers]
        B --> C[LLM Dual-Tier Extractor]
        C --> D[Tag Taxonomy Resolver]
    end

    subgraph Storage
        D --> E[FalkorDB / GraphStore: dev_graph & user_graph]
        D --> F[pgvector / VectorStore: Contextual Embeddings]
    end

    subgraph Routing Layer
        Q[Agent Query] --> AC[Audience Classifier]
        AC --> R{Has Strong Tag Signal?}
        R -->|Yes| TG[Tag Traversal on Graph]
        R -->|No| VS[Vector Similarity Fallback]
        TG --> G{Coverage Gaps?}
        G -->|Yes| VS
        G -->|No| RK[Scorer & Ranker]
        VS --> RK
        RK --> OUT[Structured Subgraph Result]
    end
```

---

## stack

- [falkordb](https://falkordb.com) for the graph layer
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity fallback
- [featherless.ai](https://featherless.ai) + [litellm](https://github.com/BerriAI/litellm) for inference
- python sdk + hosted api + cli

---

## ci

`.github/workflows/ci.yml` has two jobs. `test` runs the unit suite on the supported python
floor and ceiling, then gates on the eval hit rate. `integration` brings falkordb and pgvector
up from `docker-compose.yml` and runs the live-backend suite, which is why it only runs on
`main` and manual dispatch. Pull requests run `test` only.

the integration job is honest by construction: it installs the client libraries (the stores
silently fall back to memory without them) and fails on any skip, so a live test can never
quietly turn into a green build. it covers 24 tests across falkordb and pgvector, including
audience isolation and a fresh-database regression.

locally, `pytest -q -m "not integration"` runs the hermetic unit suite; the integration suite
needs `docker compose up -d` first. jobs run on **tenki runners** (`tenki-standard-small-2c-4g`),
which are a drop-in replacement for github-hosted runners: only the `runs-on` value differs.

one-time setup, done once per owner rather than per push: install the tenki github app for the
account that hosts this repository and grant it the repo (tenki dashboard -> runners -> github
connection). the repository currently lives at `arsoninstigator/retelldb` and moves back to the
`retelldb` org later, so no workflow value is org specific.
until that is done, jobs sit in the queue waiting for a runner with the tenki label instead of
failing.

## known issues & limitations

- **one eval miss left**: the golden query `dev_ingest_example` retrieves the example node
  (`Ingesting a Document - Example 1 (python)`) rather than its parent section, which the
  golden set expects by title. evaluation top-k hit rate is **92.3%** (12/13) on the
  `memory` backend, with 100.0% retrieval path and 100.0% audience routing accuracy. the
  known issue from the previous round (multi-tag overlap losing to hop proximity in the auth
  cluster) is fixed by weighted traversal plus duplicate-title collapsing.
- **generated ontologies need review**: `retelldb taxonomy generate` produces a taxonomy
  file, not a graph mutation. alias confidence is a model judgement, and the file is meant to
  be read before `RETELLDB_TAXONOMY_PATH` points at it.
- **citation verification needs source text**: without a `SourceVerifier` or a live file
  resolver, citations come back with `verified: null` rather than a claim.
- **persona coverage**: personas are only enforced where content was stamped with them at
  ingest. unstamped content is shared by default, which is a deliberate default but does mean
  a caller must decide visibility when it writes.

## v2

v2 adds a live call layer.

you open a call, mumble something half-formed like "how do i... the auth thing... with the token refresh", and the agent infers your intent from the audio in real time and traverses the graph to find what you actually meant. no need to type a precise query.

built on livekit and assemblyai.

