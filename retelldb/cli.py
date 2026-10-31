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


@taxonomy.command("generate")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--audience", "-a", default="developer", type=click.Choice(["developer", "user"]))
@click.option("--out", "-o", default="taxonomy.yaml", type=click.Path(path_type=Path),
              help="Where to write the reviewed taxonomy.")
@click.option("--provider", default="adaption", type=click.Choice(["adaption", "openai-compat"]),
              help="Where the taxonomy is generated. openai-compat covers any chat-completions endpoint.")
@click.option("--model", default=None, help="Generator model id, if the provider needs one.")
@click.option("--base-url", default=None, help="Override the Adaption Labs API base URL.")
@click.option("--batch-size", default=60, type=int, help="Tags per generation request.")
@click.option("--dry-run", is_flag=True, help="Collect tags and print the prompt; call nothing.")
def taxonomy_generate(paths, audience, out, provider, model, base_url, batch_size, dry_run):
    """
    Generate a tag taxonomy from the raw tags in a set of documents.

    The files are ingested into an in-memory graph to discover the tags the corpus
    actually produces. Nothing is written to a live graph and nothing is aliased until
    you point the config at the written file.
    """
    from retelldb.wrangling.generator import (
        AdaptionTransport,
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
        transport = AdaptionTransport(
            base_url=base_url or "https://api.adaptionlabs.ai", model=model
        )
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
