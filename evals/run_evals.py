#!/usr/bin/env python3
"""
RetellDB Evaluation Framework
==============================
Ingests the corpus, runs golden queries, and produces a metrics report.

Usage:
    python evals/run_evals.py                      # runs against memory backend (no Docker needed)
    python evals/run_evals.py --backend falkordb   # runs against live compose backends
    python evals/run_evals.py --output report.md   # write report to a file

Metrics produced:
  - retrieval_path_accuracy   : fraction of queries where retrieval_path matches expected
  - top_k_hit_rate            : fraction of queries where at least one expected title is in top-k results
  - audience_routing_accuracy : fraction of queries where the correct audience graph was selected
  - latency_p50_ms / p95_ms   : per-query latency percentiles
"""

import argparse
import os
import sys
import time
import yaml
import json
import textwrap
import statistics
import pathlib
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVALS_DIR = pathlib.Path(__file__).parent
CORPUS_DIR = EVALS_DIR / "corpus"
GOLDEN_QUERIES_PATH = EVALS_DIR / "golden_queries.yaml"


# ---------------------------------------------------------------------------
# Build a RetellDB stack (pipeline + engine) wired to the chosen backend
# ---------------------------------------------------------------------------


def _build_stack(backend: str, embedding_dim: int = 384):
    """Return (pipeline, engine, config) for the chosen backend."""
    from retelldb.config import RetellDBConfig, StorageConfig, InferenceConfig
    from retelldb.ingestion.pipeline import IngestionPipeline
    from retelldb.routing.engine import RetrievalEngine

    storage_cfg = StorageConfig(
        backend=backend,
        embedding_dimension=embedding_dim,
        dev_graph_name="eval_dev_graph",
        user_graph_name="eval_user_graph",
    )
    cfg = RetellDBConfig(storage=storage_cfg)

    if backend == "falkordb":
        from retelldb.storage.falkordb_store import FalkorDBStore
        from retelldb.storage.pgvector_store import PGVectorStore
        graph_store = FalkorDBStore(config=storage_cfg)
        vector_store = PGVectorStore(config=storage_cfg)
    else:
        from retelldb.storage.memory_store import MemoryGraphStore, MemoryVectorStore
        graph_store = MemoryGraphStore()
        vector_store = MemoryVectorStore()

    pipeline = IngestionPipeline(graph_store=graph_store, vector_store=vector_store, config=cfg)
    engine = RetrievalEngine(graph_store=graph_store, vector_store=vector_store, config=cfg)
    return pipeline, engine, cfg


# ---------------------------------------------------------------------------
# Ingest all corpus documents
# ---------------------------------------------------------------------------


def ingest_corpus(pipeline, verbose: bool = False) -> dict:
    """Ingest all markdown files in evals/corpus/ and return a summary."""
    from retelldb.models.query import IngestDocumentFormat
    from retelldb.models.schema import AudienceType

    results = []
    for doc_path in sorted(CORPUS_DIR.glob("*.md")):
        content = doc_path.read_text(encoding="utf-8")

        # Parse frontmatter audience field
        audience = AudienceType.DEVELOPER
        if content.startswith("---"):
            fm_end = content.find("---", 3)
            if fm_end > 0:
                fm_text = content[3:fm_end]
                for line in fm_text.splitlines():
                    if line.strip().startswith("audience:"):
                        val = line.split(":", 1)[1].strip().strip('"').strip("'")
                        if val == "user":
                            audience = AudienceType.USER
                        break

        t0 = time.perf_counter()
        resp = pipeline.ingest(
            content=content,
            format=IngestDocumentFormat.MARKDOWN,
            audience=audience,
            source_uri=f"evals://corpus/{doc_path.name}",
        )
        elapsed = (time.perf_counter() - t0) * 1000

        results.append({
            "file": doc_path.name,
            "audience": audience.value,
            "nodes_created": resp.nodes_created,
            "edges_created": resp.edges_created,
            "tags_created": resp.tags_created,
            "elapsed_ms": round(elapsed, 1),
        })
        if verbose:
            print(
                f"  ✓ {doc_path.name:<40} audience={audience.value:<10} "
                f"nodes={resp.nodes_created}  edges={resp.edges_created}  "
                f"tags={resp.tags_created}  {elapsed:.0f}ms"
            )

    return {"ingested": results, "total_docs": len(results)}


# ---------------------------------------------------------------------------
# Run golden queries
# ---------------------------------------------------------------------------


def run_queries(engine, cfg, golden_queries: list, top_k: int = 5, verbose: bool = False) -> list:
    """Execute every golden query and return raw result records."""
    from retelldb.models.query import QueryRequest
    from retelldb.models.schema import AudienceType
    from retelldb.routing.engine import QueryStrategy

    results = []

    for gq in golden_queries:
        audience_hint = None
        if gq.get("audience") == "developer":
            audience_hint = AudienceType.DEVELOPER
        elif gq.get("audience") == "user":
            audience_hint = AudienceType.USER

        request = QueryRequest(
            query=gq["query"],
            audience=audience_hint,
            top_k=top_k,
            include_edges=False,
        )

        t0 = time.perf_counter()
        result = engine.retrieve(request)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_titles = [n.node.title for n in result.nodes]
        retrieved_path = result.retrieval_path.value  # e.g. "tag_traversal"

        # Determine which graph was targeted
        dev_graph = cfg.storage.dev_graph_name
        user_graph = cfg.storage.user_graph_name
        if result.target_graph == dev_graph:
            actual_audience_graph = "dev_graph"
        elif result.target_graph == user_graph:
            actual_audience_graph = "user_graph"
        else:
            actual_audience_graph = result.target_graph

        expected_titles = gq.get("expected_titles", [])
        expected_path = gq.get("expected_retrieval_path", "")
        expected_graph = gq.get("expected_audience_graph", "")

        # Compute per-query pass/fail
        hit = any(t in retrieved_titles for t in expected_titles)
        path_correct = retrieved_path == expected_path
        graph_correct = actual_audience_graph == expected_graph

        record = {
            "id": gq["id"],
            "query": gq["query"],
            "audience_hint": gq.get("audience"),
            "expected_titles": expected_titles,
            "expected_path": expected_path,
            "expected_graph": expected_graph,
            "retrieved_titles": retrieved_titles,
            "retrieved_path": retrieved_path,
            "actual_graph": actual_audience_graph,
            "hit": hit,
            "path_correct": path_correct,
            "graph_correct": graph_correct,
            "latency_ms": round(latency_ms, 2),
            "confidence_signals": result.confidence_signals,
        }
        results.append(record)

        if verbose:
            status = "✓" if (hit and path_correct and graph_correct) else "✗"
            print(
                f"  {status} [{gq['id']}]  path={retrieved_path:<15}  "
                f"hit={hit}  graph={graph_correct}  {latency_ms:.0f}ms"
            )
            if not hit:
                print(f"      expected: {expected_titles}")
                print(f"      got:      {retrieved_titles}")

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(records: list) -> dict:
    if not records:
        return {}

    n = len(records)
    latencies = [r["latency_ms"] for r in records]
    latencies_sorted = sorted(latencies)

    def percentile(data, p):
        idx = max(0, int(len(data) * p / 100) - 1)
        return data[idx]

    return {
        "total_queries": n,
        "top_k_hit_rate": round(sum(r["hit"] for r in records) / n, 4),
        "retrieval_path_accuracy": round(sum(r["path_correct"] for r in records) / n, 4),
        "audience_routing_accuracy": round(sum(r["graph_correct"] for r in records) / n, 4),
        "latency_p50_ms": round(percentile(latencies_sorted, 50), 2),
        "latency_p95_ms": round(percentile(latencies_sorted, 95), 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(
    ingest_summary: dict,
    records: list,
    metrics: dict,
    backend: str,
    top_k: int,
    embedding_mode: str = "token-projection fallback",
) -> str:
    lines = []

    lines.append("# RetellDB Evaluation Report")
    lines.append("")
    lines.append(f"**Backend**: `{backend}`  |  **top_k**: {top_k}")
    lines.append(f"**Embedding Mode**: `{embedding_mode}`")
    lines.append(f"**Corpus docs**: {ingest_summary['total_docs']}")
    lines.append("")

    # Summary metrics table
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total queries | {metrics['total_queries']} |")
    lines.append(f"| Top-k hit rate | **{metrics['top_k_hit_rate']:.1%}** |")
    lines.append(f"| Retrieval path accuracy | **{metrics['retrieval_path_accuracy']:.1%}** |")
    lines.append(f"| Audience routing accuracy | **{metrics['audience_routing_accuracy']:.1%}** |")
    lines.append(f"| Latency p50 | {metrics['latency_p50_ms']} ms |")
    lines.append(f"| Latency p95 | {metrics['latency_p95_ms']} ms |")
    lines.append(f"| Latency mean | {metrics['latency_mean_ms']} ms |")
    lines.append("")

    # Ingestion summary
    lines.append("## Ingestion Summary")
    lines.append("")
    lines.append("| File | Audience | Nodes | Edges | Tags | Time (ms) |")
    lines.append("|---|---|---|---|---|---|")
    for item in ingest_summary["ingested"]:
        lines.append(
            f"| {item['file']} | {item['audience']} | {item['nodes_created']} "
            f"| {item['edges_created']} | {item['tags_created']} | {item['elapsed_ms']} |"
        )
    lines.append("")

    # Per-query results
    lines.append("## Query Results")
    lines.append("")
    lines.append(
        "| # | ID | Path Correct | Hit | Graph Correct | Retrieved Path | Retrieved Titles | Latency (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(records, 1):
        path_ok = "✓" if r["path_correct"] else "✗"
        hit_ok = "✓" if r["hit"] else "✗"
        graph_ok = "✓" if r["graph_correct"] else "✓" if r["graph_correct"] else "✗"
        titles_str = ", ".join(r["retrieved_titles"][:3]) or "—"
        lines.append(
            f"| {i} | `{r['id']}` | {path_ok} | {hit_ok} | "
            f"{'✓' if r['graph_correct'] else '✗'} | `{r['retrieved_path']}` "
            f"| {titles_str} | {r['latency_ms']} |"
        )
    lines.append("")

    # Failures detail
    failures = [r for r in records if not (r["hit"] and r["path_correct"] and r["graph_correct"])]
    if failures:
        lines.append("## Failure Details")
        lines.append("")
        for r in failures:
            lines.append(f"### `{r['id']}`")
            lines.append(f"**Query**: {r['query']}")
            lines.append(f"- Expected path: `{r['expected_path']}` → got `{r['retrieved_path']}`  {'✓' if r['path_correct'] else '✗'}")
            lines.append(f"- Expected graph: `{r['expected_graph']}` → got `{r['actual_graph']}`  {'✓' if r['graph_correct'] else '✗'}")
            lines.append(f"- Expected titles: {r['expected_titles']}")
            lines.append(f"- Retrieved titles: {r['retrieved_titles']}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="RetellDB evaluation runner")
    parser.add_argument(
        "--backend",
        choices=["memory", "falkordb"],
        default="memory",
        help="Storage backend to use. 'falkordb' requires docker compose up.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of nodes to retrieve per query."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the markdown report (default: print to stdout).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-item progress output."
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # 1. Build stack
    print("Building storage stack...")
    pipeline, engine, cfg = _build_stack(backend=args.backend)

    # Detect embedding mode
    status = engine.embedder.get_status()
    if status.get("is_neural"):
        embedding_mode = f"neural ({status.get('model') or 'dense'})"
    else:
        info = status.get("provider_info", "token-projection fallback")
        embedding_mode = f"token-projection fallback ({info})"

    print(f"\n=== RetellDB Eval — backend={args.backend} top_k={args.top_k} mode={embedding_mode} ===\n")

    # 2. Ingest corpus
    print(f"\nIngesting corpus from: {CORPUS_DIR}")
    ingest_summary = ingest_corpus(pipeline, verbose=verbose)
    print(
        f"  Total: {ingest_summary['total_docs']} docs ingested"
    )

    # 3. Load golden queries
    golden_data = yaml.safe_load(GOLDEN_QUERIES_PATH.read_text(encoding="utf-8"))
    golden_queries = golden_data["queries"]
    print(f"\nRunning {len(golden_queries)} golden queries...")

    # 4. Run queries
    records = run_queries(engine, cfg, golden_queries, top_k=args.top_k, verbose=verbose)

    # 5. Compute metrics
    metrics = compute_metrics(records)

    # 6. Render report
    report = render_report(
        ingest_summary=ingest_summary,
        records=records,
        metrics=metrics,
        backend=args.backend,
        top_k=args.top_k,
        embedding_mode=embedding_mode,
    )

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nReport written to: {out_path.resolve()}")
    else:
        print("\n" + "=" * 72)
        print(report)
        print("=" * 72)

    # Print summary line
    print(
        f"\n✓ Eval complete  "
        f"hit={metrics['top_k_hit_rate']:.1%}  "
        f"path_acc={metrics['retrieval_path_accuracy']:.1%}  "
        f"audience_acc={metrics['audience_routing_accuracy']:.1%}  "
        f"p50={metrics['latency_p50_ms']}ms  "
        f"p95={metrics['latency_p95_ms']}ms"
    )

    # Exit with non-zero if more than 30% failures
    if metrics["top_k_hit_rate"] < 0.70:
        print("\n⚠ Top-k hit rate below 70% threshold — eval FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
