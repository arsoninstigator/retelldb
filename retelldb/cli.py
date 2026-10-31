"""
RetellDB Command-Line Interface.
"""

import click
import json
from pathlib import Path
from retelldb.client import RetellDB
from retelldb.models.schema import AudienceType
from retelldb.models.query import IngestDocumentFormat


@click.group()
def main():
    """RetellDB: Graph-powered retrieval for agents that understand your docs."""
    pass


@main.command()
@click.option("--host", default="0.0.0.0", help="Host interface to bind.")
@click.option("--port", default=8000, type=int, help="Port to listen on.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for dev.")
def serve(host: str, port: int, reload: bool):
    """Start the RetellDB hosted API server."""
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException("uvicorn is not installed; pip install 'retelldb[all]'")
    click.echo(f"Starting RetellDB API server on http://{host}:{port}...")
    uvicorn.run("retelldb.server.app:app", host=host, port=port, reload=reload)


@main.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "-f", "doc_format", default="markdown", 
              type=click.Choice(["markdown", "mdx", "openapi", "asyncapi", "html", "notion"]))
@click.option("--audience", "-a", default="developer", type=click.Choice(["developer", "user"]))
@click.option("--title", "-t", default=None, help="Optional document title override.")
@click.option("--url", default=None, help="Remote RetellDB server URL (e.g. http://localhost:8000).")
def ingest(file_path: Path, doc_format: str, audience: str, title: str, url: str):
    """Ingest a documentation file into RetellDB."""
    content = file_path.read_text(encoding="utf-8")
    client = RetellDB(base_url=url)
    
    click.echo(f"Ingesting '{file_path.name}' into {audience} graph...")
    resp = client.ingest(
        content=content,
        format=IngestDocumentFormat(doc_format),
        audience=AudienceType(audience),
        title=title or file_path.stem,
        source_uri=str(file_path)
    )
    click.echo(f"Ingestion complete in {resp.execution_time_ms}ms:")
    click.echo(f" - Graph: {resp.graph}")
    click.echo(f" - Nodes created: {resp.nodes_created}")
    click.echo(f" - Edges created: {resp.edges_created}")
    click.echo(f" - Tags created: {resp.tags_created}")


@main.command()
@click.argument("query_text")
@click.option("--audience", "-a", default=None, type=click.Choice(["developer", "user"]), help="Audience override.")
@click.option("--max-hops", "-h", default=3, type=int, help="Maximum graph traversal hops.")
@click.option("--top-k", "-k", default=5, type=int, help="Maximum number of nodes to return.")
@click.option("--url", default=None, help="Remote RetellDB server URL.")
def query(query_text: str, audience: str, max_hops: int, top_k: int, url: str):
    """Query RetellDB documentation graph."""
    client = RetellDB(base_url=url)
    aud_enum = AudienceType(audience) if audience else None

    result = client.query(
        query=query_text,
        audience=aud_enum,
        max_hops=max_hops,
        top_k=top_k,
        include_edges=True
    )

    click.echo(f"\n--- Query: '{result.query}' ---")
    click.echo(f"Target Graph: {result.target_graph}")
    click.echo(f"Retrieval Path: {result.retrieval_path.value}")
    click.echo(f"Execution Time: {result.execution_time_ms}ms")
    click.echo(f"Confidence Signals: {result.confidence_signals}\n")

    click.echo(f"Retrieved Nodes ({len(result.nodes)}):")
    for idx, sn in enumerate(result.nodes, 1):
        click.echo(f"  [{idx}] ({sn.node.type.value}) {sn.node.title} - Score: {sn.score} (via {sn.retrieval_source})")
        if sn.node.tags:
            click.echo(f"      Tags: {', '.join(sn.node.tags)}")

    if result.edges:
        click.echo(f"\nConnecting Edges ({len(result.edges)}):")
        for e in result.edges[:10]:
            click.echo(f"  ({e.source_id}) --[{e.type.value}]--> ({e.target_id})")

    if result.traversal_chains:
        click.echo(f"\nTraversal Chains:")
        for chain in result.traversal_chains:
            click.echo(f"  {' -> '.join(chain)}")


@main.command()
@click.option("--url", default=None, help="Remote RetellDB server URL.")
def status(url: str):
    """Get status of dev and user graphs."""
    client = RetellDB(base_url=url)
    stat = client.status()
    click.echo(json.dumps(stat.model_dump(), indent=2))




@main.group()
def taxonomy():
    """Build and inspect the canonical tag ontology."""
    pass


def _latent_pass(path, out, model=None, threshold=None, max_per_tag=None, apply=False):
    """
    Embed a taxonomy's tags and write latent association proposals.

    Shared by `taxonomy associate` and `generate --associate` so the chained build and the
    standalone pass cannot drift. The embedder is imported here, not at module scope: it is
    build-time only and the runtime never loads it.
    """
    from retelldb.wrangling.generator import taxonomy_to_yaml
    from retelldb.wrangling.latent import (
        DEFAULT_MAX_PER_TAG,
        MIN_SIMILARITY,
        LatentError,
        apply_proposals,
        embed_taxonomy,
        propose_associations,
        proposals_to_yaml,
    )
    from retelldb.wrangling.taxonomy import TagTaxonomy

    taxonomy = TagTaxonomy.from_file(path)
    if not taxonomy.tags:
        raise click.ClickException(f"{path} has no canonical tags to associate")

    from retelldb.buildtime.featherless import EmbeddingUnavailable, FeatherlessEmbedder

    try:
        embedder = FeatherlessEmbedder(model=model) if model else FeatherlessEmbedder()
        vectors = embed_taxonomy(embedder, taxonomy)
        report = propose_associations(
            taxonomy,
            vectors,
            threshold=threshold if threshold is not None else MIN_SIMILARITY,
            max_per_tag=max_per_tag or DEFAULT_MAX_PER_TAG,
            embedder=embedder.name,
        )
    except (LatentError, EmbeddingUnavailable, RuntimeError) as exc:
        raise click.ClickException(str(exc))

    out.write_text(proposals_to_yaml(report), encoding="utf-8")
    click.echo(
        f"Embedded {report.tags_embedded} tags with {report.embedder} "
        f"({report.dimension}-dim, threshold {report.threshold})."
    )
    click.echo(
        f"{len(report.proposals)} proposals, {report.below_threshold} pairs below threshold, "
        f"{report.skipped_already_related} already configured, "
        f"{report.blocked_by_exclusion} blocked by an exclusion."
    )
    click.echo(
        f"This corpus sits at median {report.similarity_median} and p90 "
        f"{report.similarity_p90} across {report.pairs_compared} pairs."
    )

    if apply:
        apply_proposals(taxonomy, report)
        path.write_text(taxonomy_to_yaml(taxonomy), encoding="utf-8")
        click.echo(f"Applied {len(report.proposals)} associations to {path}.")
    else:
        click.echo(f"Wrote {out}. Review it, then re-run with --apply to accept.")
    return report


@taxonomy.command("generate")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--audience", "-a", default="developer", type=click.Choice(["developer", "user"]))
@click.option("--out", "-o", default="taxonomy.yaml", type=click.Path(path_type=Path),
              help="Where to write the reviewed taxonomy.")
@click.option("--provider", default="adaption", type=click.Choice(["adaption", "openai-compat"]),
              help="Where the taxonomy is generated. openai-compat covers any chat-completions endpoint.")
@click.option("--model", default=None, help="Generator model id, if the provider needs one.")
@click.option("--base-url", default=None, help="Override the Adaption Labs API base URL.")
@click.option("--batch-size", default=None, type=int,
              help="Tags per generation request. Defaults to the generator's safe batch size.")
@click.option("--associate", is_flag=True,
              help="After writing the taxonomy, run the latent pass and write its proposals.")
@click.option("--dry-run", is_flag=True, help="Collect tags and print the prompt; call nothing.")
def taxonomy_generate(paths, audience, out, provider, model, base_url, batch_size, associate,
                      dry_run):
    """
    Generate a tag taxonomy from the raw tags in a set of documents.

    The files are ingested into an in-memory graph to discover the tags the corpus
    actually produces. Nothing is written to a live graph and nothing is aliased until
    you point the config at the written file.
    """
    from retelldb.wrangling.generator import (
        DEFAULT_BATCH_SIZE,
        GenerationError,
        build_prompt,
        collect_inventory,
        generate_taxonomy,
        taxonomy_to_yaml,
    )

    if not paths:
        raise click.UsageError("give at least one document path or directory")

    documents = []
    for path in paths:
        if path.is_dir():
            documents.extend(sorted(p for p in path.rglob("*") if p.suffix in (".md", ".mdx")))
        else:
            documents.append(path)
    if not documents:
        raise click.UsageError("no markdown documents found in the given paths")

    client = RetellDB(use_memory=True)
    graph_name = (
        client.config.storage.dev_graph_name
        if audience == "developer"
        else client.config.storage.user_graph_name
    )

    for path in documents:
        client.ingest(
            content=path.read_text(encoding="utf-8"),
            format=IngestDocumentFormat("markdown" if path.suffix in (".md", ".mdx") else "html"),
            audience=AudienceType(audience),
            title=path.stem,
            source_uri=str(path),
        )

    tag_nodes = client.graph_store.get_tag_nodes(graph_name)
    inventory = collect_inventory([t.title for t in tag_nodes])

    batch_size = batch_size or DEFAULT_BATCH_SIZE
    click.echo(f"Collected {len(inventory.tags)} raw tags from {len(documents)} document(s).")
    if dry_run:
        click.echo(build_prompt(inventory.tags[:batch_size], inventory.snippets))
        return

    if provider == "openai-compat":
        from retelldb.wrangling.generator import OpenAICompatTransport

        transport = OpenAICompatTransport(
            base_url=base_url or "https://api.featherless.ai/v1",
            model=model or "Qwen/Qwen2.5-72B-Instruct",
        )
    else:
        # build-time only: the transport itself refuses to run in a production deployment
        from retelldb.buildtime.adaption import AdaptionTransport

        transport = AdaptionTransport(base_url=base_url, model=model)
    try:
        generated, report = generate_taxonomy(
            inventory, transport, batch_size=batch_size
        )
    except GenerationError as exc:
        raise click.ClickException(str(exc))

    out.write_text(taxonomy_to_yaml(generated), encoding="utf-8")
    click.echo(f"Wrote {out} with {report['canonical_count']} canonical tags "
               f"and {report['alias_count']} aliases.")
    uncovered = report.get("uncovered_raw_tags") or []
    if uncovered:
        click.echo(f"Uncovered raw tags ({len(uncovered)}): {', '.join(uncovered[:20])}")
    dangling = report.get("dangling_references") or []
    if dangling:
        click.echo(f"References with no canonical tag in this file: {', '.join(dangling)}")
    collisions = report.get("alias_collisions") or []
    if collisions:
        click.echo(f"Alias collisions to review: {', '.join(collisions)}")
    ambiguous = report.get("ambiguous") or []
    if ambiguous:
        click.echo(f"Ambiguous tags left unresolved: {len(ambiguous)}")
    click.echo("Review the file, then set RETELLDB_TAXONOMY_PATH to use it.")
    if associate:
        proposals_path = out.with_name(f"{out.stem}.associations.yaml")
        _latent_pass(out, proposals_path)


@taxonomy.command("associate")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--model", default=None,
              help="Embedding model id (Featherless Qwen3-Embedding family).")
@click.option("--threshold", default=None, type=float,
              help="Minimum cosine similarity to propose. Defaults to the uncertain tier.")
@click.option("--max-per-tag", default=None, type=int,
              help="Neighbours kept per tag, strongest first.")
@click.option("--out", "-o", default=Path("associations.yaml"), type=click.Path(path_type=Path),
              help="Where to write the proposals for review.")
@click.option("--apply", is_flag=True,
              help="Accept every proposal into the taxonomy. Configured weights still win.")
def taxonomy_associate(path, model, threshold, max_per_tag, out, apply):
    """
    Propose association edges from latent tag similarity.

    Embeddings propose, a human accepts: the run writes proposals with their scores as
    provenance and touches no graph. --apply writes the accepted strengths into the
    taxonomy file after you have read them.
    """
    _latent_pass(path, out, model=model, threshold=threshold,
                 max_per_tag=max_per_tag, apply=apply)


@taxonomy.command("validate")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def taxonomy_validate(path):
    """Load a taxonomy file and report what it declares."""
    from retelldb.wrangling.taxonomy import TagTaxonomy

    loaded = TagTaxonomy.from_file(path)
    canonicals = loaded.canonical_names()
    alias_count = sum(len(t.aliases) for t in loaded.tags)
    dangling = sorted(
        {ref for tag in loaded.tags for ref in list(tag.parents) + list(tag.related)}
        - set(canonicals)
    )
    click.echo(f"{path}: {len(canonicals)} canonical tags, {alias_count} aliases")
    if dangling:
        click.echo(f"References with no canonical tag: {', '.join(dangling)}")

if __name__ == "__main__":
    main()
