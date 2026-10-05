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

logger = logging.getLogger("retelldb.pipeline")


class IngestionPipeline:
    """End-to-end documentation ingestion pipeline for RetellDB."""

    def __init__(
        self,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        config: Optional[RetellDBConfig] = None
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.config = config or default_config
        
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

    def ingest(
        self,
        content: str,
        format: IngestDocumentFormat,
        audience: AudienceType = AudienceType.DEVELOPER,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
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

            # 6. Tag Taxonomy Resolver (Create first-class Tag nodes & tagged_as edges for this audience)
            tag_nodes, tag_edges = self._resolve_tag_taxonomy(aud_nodes, aud)

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

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return IngestResponse(
            success=True,
            nodes_created=total_nodes_created,
            edges_created=total_edges_created,
            tags_created=total_tags_created,
            graph=",".join(target_graphs_written),
            execution_time_ms=round(elapsed_ms, 2)
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
    ) -> (List[GraphNode], List[GraphEdge]):
        """Ensure tags are first-class nodes scoped to their audience namespace with deterministic IDs."""
        tag_map: Dict[str, GraphNode] = {}
        tag_edges: List[GraphEdge] = []

        for node in nodes:
            for tag_name in node.tags:
                clean_tag = tag_name.lower().strip()
                if not clean_tag:
                    continue

                tag_id = f"tag:{audience.value}:{clean_tag}"
                if clean_tag not in tag_map:
                    tag_node = GraphNode(
                        id=tag_id,
                        title=clean_tag,
                        type=NodeType.TAG,
                        content=f"Semantic tag node: {clean_tag}",
                        audience=audience,
                        tags=[clean_tag],
                        metadata={"is_taxonomy_tag": True}
                    )
                    tag_map[clean_tag] = tag_node

                edge_id = generate_edge_id(node.id, tag_id, EdgeType.TAGGED_AS.value)
                tag_edges.append(
                    GraphEdge(
                        id=edge_id,
                        source_id=node.id,
                        target_id=tag_id,
                        type=EdgeType.TAGGED_AS,
                        weight=1.0
                    )
                )

        return list(tag_map.values()), tag_edges
