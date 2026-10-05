"""
RetellDB Command-Line Interface.
"""

import click
import json
import uvicorn
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


if __name__ == "__main__":
    main()
