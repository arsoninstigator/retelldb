"""
Ingestion Pipeline orchestrator.
Normalizes multi-format documents, executes dual-tier LLM extraction,
resolves tag taxonomy nodes, computes contextual embeddings, and writes to graph & vector stores.
Ensures strict dual-graph audience isolation and idempotent source-URI upserts.
"""

import time
import logging
from typing import Optional, Dict, Any, List, Set
from retelldb.models.schema import (
    NodeType,
    EdgeType,
    AudienceType,
    GraphNode,
    GraphEdge,
    SourceRef,
    generate_edge_id,
)
from retelldb.models.query import IngestDocumentFormat, IngestResponse
from retelldb.ingestion.normalizers.base import BaseNormalizer
from retelldb.ingestion.normalizers.markdown import MarkdownNormalizer
from retelldb.ingestion.normalizers.openapi import OpenAPINormalizer
from retelldb.ingestion.normalizers.html import HTMLNormalizer
from retelldb.ingestion.normalizers.notion import NotionNormalizer
from retelldb.ingestion.extractor import LLMExtractor
from retelldb.storage.base import BaseGraphStore, BaseVectorStore
from retelldb.storage.embedder import ContextualEmbedder
from retelldb.config import RetellDBConfig, default_config
from retelldb.wrangling.taxonomy import (
    ResolutionRule,
    TagResolution,
    TagTaxonomy,
    taxonomy_from_config,
)

# content type recorded on a citation when the normalizer did not provide one
FORMAT_CONTENT_TYPES = {
    IngestDocumentFormat.MARKDOWN: "markdown",
    IngestDocumentFormat.MDX: "markdown",
    IngestDocumentFormat.OPENAPI: "openapi",
    IngestDocumentFormat.ASYNCAPI: "asyncapi",
    IngestDocumentFormat.NOTION: "notion",
    IngestDocumentFormat.HTML: "html",
}

logger = logging.getLogger("retelldb.pipeline")


class IngestionPipeline:
    """End-to-end documentation ingestion pipeline for RetellDB."""

    def __init__(
        self,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        config: Optional[RetellDBConfig] = None,
        taxonomy: Optional[TagTaxonomy] = None
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.config = config or default_config
        self.taxonomy = taxonomy or self._load_taxonomy()

        self.extractor = LLMExtractor(self.config.inference)
        self.embedder = ContextualEmbedder(self.config.inference)
        
        self.normalizers: Dict[IngestDocumentFormat, BaseNormalizer] = {
            IngestDocumentFormat.MARKDOWN: MarkdownNormalizer(),
            IngestDocumentFormat.MDX: MarkdownNormalizer(),
            IngestDocumentFormat.OPENAPI: OpenAPINormalizer(),
            IngestDocumentFormat.ASYNCAPI: OpenAPINormalizer(),
            IngestDocumentFormat.HTML: HTMLNormalizer(),
            IngestDocumentFormat.NOTION: NotionNormalizer(),
        }

    def _load_taxonomy(self) -> TagTaxonomy:
        """Load the configured tag ontology, or an empty one (no aliasing)."""
        return taxonomy_from_config(self.config)

    def ingest(
        self,
        content: str,
        format: IngestDocumentFormat,
        audience: AudienceType = AudienceType.DEVELOPER,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        personas: Optional[List[str]] = None,
        sensitivity: str = "shared"
    ) -> IngestResponse:
        """Run full ingestion pipeline with strict audience isolation and source upsert."""
        t_start = time.perf_counter()

        # 1. Select Normalizer & Parse Document
        normalizer = self.normalizers.get(format, self.normalizers[IngestDocumentFormat.MARKDOWN])
        norm_doc = normalizer.normalize(
            content=content,
            audience=audience,
            title=title,
            source_uri=source_uri,
            **(metadata or {})
        )

        # 2. Fast Tier LLM Extraction: refine nodes and normalize tags
        refined_nodes = self.extractor.refine_nodes(norm_doc.nodes)

        # 2b. Every node must be able to answer "where did you come from": normalizers
        # that cannot compute line ranges still record the file and content type.
        content_type = FORMAT_CONTENT_TYPES.get(format, "text")
        for node in refined_nodes:
            if not node.sources:
                node.sources = [
                    SourceRef(
                        uri=node.source_uri or source_uri,
                        path=node.source_uri or source_uri,
                        content_type=content_type,
                        excerpt=node.content[:200],
                    )
                ]

        # 2c. Perspective: stamp persona visibility so scope decisions are data, not guesses
        if personas:
            normalized_personas = [p.strip().lower() for p in personas if p and p.strip()]
            for node in refined_nodes:
                node.personas = list(normalized_personas)
                node.sensitivity = sensitivity

        # 3. Strong Tier LLM Extraction: infer relationships
        final_edges = self.extractor.infer_relationships(refined_nodes, norm_doc.edges)

        # 4. Strict Audience Isolation Partitioning
        # Group nodes by their actual resolved audience
        nodes_by_audience: Dict[AudienceType, List[GraphNode]] = {
            AudienceType.DEVELOPER: [],
            AudienceType.USER: []
        }
        for n in refined_nodes:
            nodes_by_audience[n.audience].append(n)

        total_nodes_created = 0
        total_edges_created = 0
        total_tags_created = 0
        total_aliases = 0
        all_resolutions: List[TagResolution] = []
        all_ambiguous: List[str] = []
        target_graphs_written: List[str] = []

        # Process each audience partition completely independently into its isolated graph
        for aud, aud_nodes in nodes_by_audience.items():
            if not aud_nodes:
                continue

            target_graph = (
                self.config.storage.dev_graph_name 
                if aud == AudienceType.DEVELOPER 
                else self.config.storage.user_graph_name
            )
            target_graphs_written.append(target_graph)

            # Filter edges where both source and target belong to this audience partition
            aud_node_ids = {n.id for n in aud_nodes}
            aud_edges = [
                e for e in final_edges
                if e.source_id in aud_node_ids and e.target_id in aud_node_ids
            ]

            # 5. Idempotent Source-URI Upsert Handling
            effective_source = source_uri or norm_doc.document_title
            if effective_source:
                self._reconcile_source_upsert(
                    graph_name=target_graph,
                    source_uri=effective_source,
                    current_node_ids=aud_node_ids
                )

            # 6. Tag Taxonomy Resolver: canonical tags, aliases (with provenance), hierarchy
            (
                tag_nodes,
                tag_edges,
                resolutions,
                ambiguous_tags,
                alias_count,
            ) = self._resolve_tag_taxonomy(aud_nodes, aud)

            all_aud_nodes = aud_nodes + tag_nodes
            all_aud_edges = aud_edges + tag_edges

            # 7. Write Nodes and Edges to the Isolated Target Graph
            for node in all_aud_nodes:
                self.graph_store.add_node(target_graph, node)

            for edge in all_aud_edges:
                self.graph_store.add_edge(target_graph, edge)

            # 8. Generate Contextual Node Embeddings & Write to Vector Store for this Graph
            node_title_map = {n.id: n.title for n in all_aud_nodes}
            adj_map: Dict[str, List[str]] = {}
            for edge in all_aud_edges:
                adj_map.setdefault(edge.source_id, []).append(node_title_map.get(edge.target_id, ""))
                adj_map.setdefault(edge.target_id, []).append(node_title_map.get(edge.source_id, ""))

            for node in aud_nodes:  # Embed content nodes
                neighbors = [t for t in adj_map.get(node.id, []) if t]
                context_text = self.embedder.build_contextual_text(node, neighbors)
                embedding = self.embedder.embed_text(context_text)
                node.embedding = embedding
                
                self.vector_store.add_embedding(
                    graph_name=target_graph,
                    node_id=node.id,
                    embedding=embedding,
                    metadata={
                        "title": node.title,
                        "type": node.type.value,
                        "tags": node.tags,
                        "audience": node.audience.value,
                        "source_uri": node.source_uri
                    }
                )

            total_nodes_created += len(aud_nodes)
            total_edges_created += len(aud_edges)
            total_tags_created += len(tag_nodes)
            total_aliases += alias_count
            all_resolutions.extend(resolutions)
            all_ambiguous.extend(ambiguous_tags)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return IngestResponse(
            success=True,
            nodes_created=total_nodes_created,
            edges_created=total_edges_created,
            tags_created=total_tags_created,
            graph=",".join(target_graphs_written),
            execution_time_ms=round(elapsed_ms, 2),
            aliases_created=total_aliases,
            ambiguous_tags=sorted(set(all_ambiguous)),
        )

    def _reconcile_source_upsert(
        self,
        graph_name: str,
        source_uri: str,
        current_node_ids: Set[str]
    ) -> None:
        """Find stale nodes previously ingested from this source_uri and remove them."""
        existing_nodes = self.graph_store.get_nodes_by_source(graph_name, source_uri)
        stale_ids = [n.id for n in existing_nodes if n.id not in current_node_ids]
        if stale_ids:
            logger.info(f"Removing {len(stale_ids)} stale nodes for source '{source_uri}' in {graph_name}.")
            self.graph_store.delete_nodes(graph_name, stale_ids)
            self.vector_store.delete_embeddings(graph_name, stale_ids)

    def _resolve_tag_taxonomy(
        self,
        nodes: List[GraphNode],
        audience: AudienceType
    ) -> (List[GraphNode], List[GraphEdge], List[TagResolution], List[str], int):
        """
        Resolve raw tags into canonical tag nodes without rewriting the raw tags.

        Per content node:
          node --tagged_as(weight=resolution confidence)--> canonical tag node
        Per alias:
          alias tag node --alias_of(weight=confidence)--> canonical tag node
        Per declared hierarchy:
          child tag node --parent_of(1.0)--> parent tag node

        The node keeps the raw strings in `node.tags`, so the alias remains provenance
        rather than being cleaned away.
        """
        tag_nodes: Dict[str, GraphNode] = {}
        tag_edges: List[GraphEdge] = []
        resolutions: List[TagResolution] = []
        ambiguous: List[str] = []
        alias_count = 0

        def node_id(name: str) -> str:
            return f"tag:{audience.value}:{name}"

        def ensure_tag(name: str, is_alias: bool = False, canonical: Optional[str] = None) -> str:
            clean = name.lower().strip()
            if not clean:
                return ""
            tid = node_id(clean)
            if tid not in tag_nodes:
                metadata: Dict[str, Any] = {"is_canonical": not is_alias}
                tags = [clean]
                if is_alias:
                    metadata.update({"is_alias": True, "canonical": canonical})
                    tags = [canonical or clean]
                    content = f"Alias tag node: {clean} -> {canonical}"
                else:
                    content = f"Canonical tag node: {clean}"
                tag_nodes[tid] = GraphNode(
                    id=tid,
                    title=clean,
                    type=NodeType.TAG,
                    content=content,
                    audience=audience,
                    tags=tags,
                    metadata=metadata,
                )
            return tid

        # 1. Declared ontology: canonical tags, hierarchy and associations exist even
        #    before a document uses them, so query-time traversal can follow them.
        for declared in self.taxonomy.tags:
            ensure_tag(declared.name)
        for declared in self.taxonomy.tags:
            child_id = node_id(declared.normalized_name())
            for parent in declared.parents:
                parent_id = ensure_tag(parent)
                tag_edges.append(
                    GraphEdge(
                        id=generate_edge_id(child_id, parent_id, EdgeType.PARENT_OF.value),
                        source_id=child_id,
                        target_id=parent_id,
                        type=EdgeType.PARENT_OF,
                        weight=1.0,
                        metadata={"declared": True},
                    )
                )
            for related, weight in declared.related.items():
                related_id = ensure_tag(related)
                tag_edges.append(
                    GraphEdge(
                        id=generate_edge_id(child_id, related_id, EdgeType.RELATED.value),
                        source_id=child_id,
                        target_id=related_id,
                        type=EdgeType.RELATED,
                        weight=float(weight),
                        metadata={"declared": True},
                    )
                )

        # 2. Resolve each observed raw tag
        for node in nodes:
            for raw_tag in node.tags:
                resolution = self.taxonomy.resolve(raw_tag)
                resolutions.append(resolution)

                if resolution.rule == ResolutionRule.AMBIGUOUS:
                    ambiguous.append(str(raw_tag))
                    canonical = str(raw_tag).lower().strip()
                    ensure_tag(canonical)
                else:
                    canonical = resolution.canonical or str(raw_tag).lower().strip()
                    if resolution.is_alias:
                        alias_id = ensure_tag(
                            str(raw_tag).lower().strip(),
                            is_alias=True,
                            canonical=canonical,
                        )
                        canonical_id = ensure_tag(canonical)
                        alias_count += 1
                        tag_edges.append(
                            GraphEdge(
                                id=generate_edge_id(
                                    alias_id, canonical_id, EdgeType.ALIAS_OF.value
                                ),
                                source_id=alias_id,
                                target_id=canonical_id,
                                type=EdgeType.ALIAS_OF,
                                weight=resolution.confidence,
                                metadata={
                                    "rule": resolution.rule.value,
                                    "raw_tag": str(raw_tag),
                                },
                            )
                        )
                    ensure_tag(canonical)

                canonical_id = ensure_tag(canonical)
                tag_edges.append(
                    GraphEdge(
                        id=generate_edge_id(node.id, canonical_id, EdgeType.TAGGED_AS.value),
                        source_id=node.id,
                        target_id=canonical_id,
                        type=EdgeType.TAGGED_AS,
                        weight=max(0.05, min(1.0, resolution.confidence)),
                        metadata={"raw_tag": str(raw_tag), "rule": resolution.rule.value},
                    )
                )

        return list(tag_nodes.values()), tag_edges, resolutions, ambiguous, alias_count
