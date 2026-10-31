# changelog

## 0.2.4

### added

- `retelldb taxonomy associate`: latent tag similarity proposes association edges, and a human
  accepts them. embeddings never create an edge at query time, which is the memorium rule
  applied literally: the proposal carries the raw cosine as provenance, `--apply` writes the
  accepted strength, and a configured weight is never overwritten.
- exclusion pairs are never bridged. `excludes` is now a first-class field on a canonical tag,
  parsed from yaml, serialised back, respected by both proposal and apply.
- per-tag neighbour budget keeps the strongest edges, so a hub cannot claim the taxonomy.
- reports the corpus distribution (pairs compared, median, p90) so a threshold can be chosen
  against evidence instead of vibes. on the eval corpus: 32 tags, 55 proposals, median 0.549.
- `retelldb/buildtime/featherless.py`: Qwen3-Embedding over the Featherless endpoint, build-time
  only, dimension-checked, and honest about failure. no token-projection fallback, because
  token overlap is not latent space.

### notes

- the embedding lane obeys the same containment as the generator: `assert_build_time` on
  construction, and a production deployment raises instead of calling out.

## 0.2.3

### fixed

- the provider now points at the real endpoint, `https://api.prod.adaptionlabs.ai/api/v1`
  (routes carry the `/api` prefix even though the docs say `/v1`). the old
  `api.adaptionlabs.ai` host answers 404 on every path.
- every request names a model, because the endpoint rejects a request without one. the default
  is `gemma-4-31b-it`; `ADAPTIONLABS_MODEL` and `--model` override it.
- the extractor skips `reasoning` items and reads the assistant `message`, so chain-of-thought
  never reaches the json parser. it also used a private method that no longer existed after the
  build-time split.
- `normalize_base_url` accepts the host with or without `/v1` or `/api/v1`.
- batch size defaults to 12: the gateway closes a request at 60s, 16 tags took 57s and 20 timed
  out. the cli option no longer hardcodes 60.

### verified

- taxonomy generation runs end to end against the live provider on the eval corpus: 46 raw tags
  in, 32 canonical tags and 25 aliases out, clean validation.

## 0.2.2

### changed

- the model provider moved to `retelldb/buildtime/`, a package the runtime never imports.
  generation is a build step whose output is a reviewed `taxonomy.yaml`, so a production
  deployment has no path to a provider call. a test boots a clean interpreter and fails if the
  runtime import graph pulls `retelldb.buildtime` in.
- `assert_build_time()` guards construction and every completion: a production deployment
  signal (`RETELLDB_DEPLOYMENT`, `RETELLDB_ENV`, `DEPLOYMENT_ENV`, `ENVIRONMENT` set to
  `production`, `prod` or `live`) raises instead of calling out. no override flag.
- `retelldb/wrangling/generator.py` is provider agnostic and takes any object with
  `available()` and `complete(system, prompt)`, so the ontology logic carries no vendor.
- `ADAPTIONLABS_BASE_URL` overrides the endpoint; the base url is no longer hardcoded in the
  cli.

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
