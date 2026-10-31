# changelog

## 0.2.1

### fixed

- pgvector: the store registered its client adapter before creating the schema, so against a
  fresh database every call raised `vector type not found`, the error was swallowed, and
  pgvector quietly served reads and writes from the in-memory fallback. schema first,
  registration second. found by ci, covered by a regression test that builds and drops its own
  database.
- test harness: config overrides are merged instead of splatted next to defaults, so a test can
  point `pg_dsn` at another database.

### added

- ci runs the live-backend suite on tenki runners: docker-compose falkordb and pgvector inside
  the job, 24 integration tests, and a check that fails the build if any of them skip instead
  of running.
- unit job deselects integration markers, which previously collected as skips.

## 0.2.0

the tag wrangler. retrieval semantics from the tag-wrangling research are now code, not prose.

### added

- canonical tag ontology: `alias_of` and `parent_of` edges, declared associations, and raw
  tag provenance that survives resolution
- ambiguity handling: an alias mapped to two canonical tags resolves to nothing and is
  reported as ambiguous at ingest and at generation
- perspective scoping: persona and sensitivity as retrieval parameters, fail-closed, with
  `blocked` reasons in the result and no traversal through withheld nodes
- weighted traversal: one best-first implementation shared by both backends, product-of-
  weights chain scoring, edge floor pruning, `blocks` edges, cycle termination, and path
  traces in the result
- confidence tiers: `high`, `uncertain`, `needs_clarification`, `unavailable`, plus
  component breakdown, dominant dampener and next actions
- source citations: path, line range and content type per node, markdown normalizer computes
  real file line numbers (frontmatter included), optional integrity verification
- `retelldb taxonomy generate` and `retelldb taxonomy validate`, generating an ontology from
  observed tags through Adaption Labs with strict validation and honest failure
- `SourceRef`, `personas`, `sensitivity` on nodes; `persona`, `allow_shared`,
  `min_edge_weight`, `include_blocked`, `include_citations` on queries

### fixed

- duplicate titles from separate source documents are collapsed by title and type before
  ranking
- traversal scoring no longer diverges between the memory and falkordb backends
- `uvicorn` is imported lazily, so the CLI works without the server extras

### evaluation

- top-k hit rate 76.9% -> 92.3% (12/13) on the memory backend, retrieval path 100%,
  audience routing 100%

## 0.1.0

initial implementation: dual-graph ingestion, tag traversal, vector fallback, routing layer,
CLI, hosted API, integration suite and eval framework.
