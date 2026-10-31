"""
FalkorDB property graph storage adapter.
Executes native openCypher queries for graph node/edge creation, multi-hop tag traversals,
subgraph connections, source-based upsert deletion, and status reporting.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from retelldb.models.schema import GraphNode, GraphEdge, NodeType, EdgeType, AudienceType
from retelldb.storage.base import BaseGraphStore
from retelldb.storage.memory_store import MemoryGraphStore
from retelldb.config import StorageConfig, default_config

logger = logging.getLogger("retelldb.falkordb")


def _safe_node_type(raw: Any) -> NodeType:
    """Convert a raw Cypher string property to NodeType with a safe fallback."""
    try:
        return NodeType(str(raw).lower())
    except (ValueError, TypeError):
        return NodeType.CONCEPT


def _safe_audience(raw: Any) -> AudienceType:
    """Convert a raw Cypher string property to AudienceType with a safe fallback."""
    try:
        return AudienceType(str(raw).lower())
    except (ValueError, TypeError):
        return AudienceType.DEVELOPER


def _safe_tags(raw: Any) -> List[str]:
    """Coerce the 'tags' property to a list of strings regardless of how FalkorDB returned it."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str) and raw:
        # FalkorDB may serialise a list as a string like "['auth', 'tokens']"
        import ast
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except Exception:
            pass
        return [raw]
    return []


def _str_or_none(raw: Any) -> Optional[str]:
    """Return a non-empty string or None."""
    s = str(raw) if raw is not None else ""
    return s if s else None



class FalkorDBStore(BaseGraphStore):
    """Native FalkorDB backend adapter using Cypher property graph queries."""

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or default_config.storage
        self._client = None
        self._connected = False
        self._memory_fallback = None

        try:
            from falkordb import FalkorDB
            self._client = FalkorDB(
                host=self.config.falkordb_host,
                port=self.config.falkordb_port,
                password=self.config.falkordb_password
            )
            # Test connection
            self._client.connection.ping()
            self._connected = True
            logger.info("Connected directly to FalkorDB instance.")
        except Exception as e:
            logger.warning(
                f"FalkorDB connection unavailable ({e}). "
                "Operating in standalone in-memory graph mode."
            )
            self._connected = False
            self._memory_fallback = MemoryGraphStore()

    def _get_graph(self, graph_name: str):
        return self._client.select_graph(graph_name)

    def add_node(self, graph_name: str, node: GraphNode) -> None:
        if not self._connected:
            self._memory_fallback.add_node(graph_name, node)
            return

        try:
            g = self._get_graph(graph_name)
            label = node.type.value
            props = node.to_cypher_props()
            query = f"MERGE (n:{label} {{id: $id}}) SET n += $props"
            g.query(query, {"id": node.id, "props": props})
        except Exception as e:
            logger.error(f"FalkorDB add_node failed: {e}")

    def add_edge(self, graph_name: str, edge: GraphEdge) -> None:
        if not self._connected:
            self._memory_fallback.add_edge(graph_name, edge)
            return

        try:
            g = self._get_graph(graph_name)
            edge_label = edge.type.value.upper()
            props = edge.to_cypher_props()
            query = (
                f"MATCH (s {{id: $source_id}}), (t {{id: $target_id}}) "
                f"MERGE (s)-[r:{edge_label}]->(t) SET r += $props"
            )
            g.query(query, {"source_id": edge.source_id, "target_id": edge.target_id, "props": props})
        except Exception as e:
            logger.error(f"FalkorDB add_edge failed: {e}")

    def get_node(self, graph_name: str, node_id: str) -> Optional[GraphNode]:
        if not self._connected:
            return self._memory_fallback.get_node(graph_name, node_id)

        try:
            g = self._get_graph(graph_name)
            res = g.query("MATCH (n {id: $id}) RETURN n", {"id": node_id})
            if res.result_set:
                raw_node = res.result_set[0][0]
                return GraphNode.from_cypher_props(dict(raw_node.properties))
        except Exception as e:
            logger.error(f"FalkorDB get_node failed: {e}")
        return None

    def get_edges(self, graph_name: str, node_id: str) -> List[GraphEdge]:
        if not self._connected:
            return self._memory_fallback.get_edges(graph_name, node_id)

        try:
            g = self._get_graph(graph_name)
            res = g.query(
                "MATCH (s {id: $id})-[r]-(t) RETURN s.id, type(r), t.id, r",
                {"id": node_id}
            )
            edges = []
            for row in res.result_set:
                s_id, r_type, t_id, r_obj = row
                props = r_obj.properties
                # r_obj.src_node / r_obj.dest_node are internal integer node IDs;
                # use the Cypher-projected string property IDs (s_id, t_id) instead.
                edge_id = props.get("id") or f"{s_id}-{r_type}-{t_id}"
                edges.append(
                    GraphEdge(
                        id=edge_id,
                        source_id=props.get("source_id") or str(s_id),
                        target_id=props.get("target_id") or str(t_id),
                        type=EdgeType(r_type.lower()),
                        weight=float(props.get("weight", 1.0)),
                    )
                )
            return edges
        except Exception as e:
            logger.error(f"FalkorDB get_edges failed: {e}")
            return []

    def neighbors(
        self,
        graph_name: str,
        node_id: str
    ) -> List[Tuple[GraphEdge, str]]:
        """Adjacency via the same Cypher edge projection used by get_edges."""
        if not self._connected:
            return self._memory_fallback.neighbors(graph_name, node_id)

        pairs: List[Tuple[GraphEdge, str]] = []
        for edge in self.get_edges(graph_name, node_id):
            other = edge.target_id if edge.source_id == node_id else edge.source_id
            if other:
                pairs.append((edge, other))
        return pairs

    def get_tag_nodes(self, graph_name: str) -> List[GraphNode]:
        if not self._connected:
            return self._memory_fallback.get_tag_nodes(graph_name)

        try:
            g = self._get_graph(graph_name)
            res = g.query("MATCH (t:tag) RETURN t")
            nodes = []
            for row in res.result_set:
                props = row[0].properties
                nodes.append(
                    GraphNode(
                        id=props.get("id"),
                        title=props.get("title", ""),
                        type=NodeType.TAG,
                        content=props.get("content", ""),
                        tags=_safe_tags(props.get("tags", [])),
                    )
                )
            return nodes
        except Exception as e:
            logger.error(f"FalkorDB get_tag_nodes failed: {e}")
            return []

    def tag_traversal(
        self, 
        graph_name: str, 
        tag_names: List[str], 
        max_hops: int = 3
    ) -> Tuple[List[Tuple[GraphNode, int]], List[GraphEdge]]:
        if not self._connected:
            return self._memory_fallback.tag_traversal(graph_name, tag_names, max_hops)

        try:
            g = self._get_graph(graph_name)
            lower_tags = [t.lower().strip() for t in tag_names]
            query = (
                f"MATCH (t:tag) WHERE toLower(t.title) IN $tags "
                f"MATCH path = (t)-[*1..{max_hops}]-(n) "
                f"RETURN nodes(path), relationships(path), length(path)"
            )
            res = g.query(query, {"tags": lower_tags})
            
            node_map: Dict[str, Tuple[GraphNode, int]] = {}
            edge_map: Dict[str, GraphEdge] = {}

            for row in res.result_set:
                p_nodes, p_rels, path_len = row
                for idx, n in enumerate(p_nodes):
                    props = n.properties
                    nid = props.get("id")
                    hop = idx  # 0 for tag, 1 for hop 1, etc.
                    if nid:
                        if nid not in node_map or node_map[nid][1] > hop:
                            node_map[nid] = (
                                GraphNode.from_cypher_props(dict(props)),
                                hop
                            )
                for r in p_rels:
                    props = r.properties
                    # r.src_node and r.dest_node may be integer internal IDs or Node objects;
                    # extract the integer id safely for fallback edge ID construction.
                    src_raw = r.src_node
                    dst_raw = r.dest_node
                    src_int = src_raw.id if hasattr(src_raw, "id") else src_raw
                    dst_int = dst_raw.id if hasattr(dst_raw, "id") else dst_raw
                    rid = props.get("id") or f"{src_int}-{r.relation}-{dst_int}"
                    if rid not in edge_map:
                        edge_map[rid] = GraphEdge(
                            id=rid,
                            source_id=props.get("source_id") or str(src_int),
                            target_id=props.get("target_id") or str(dst_int),
                            type=EdgeType(r.relation.lower()),
                            weight=float(props.get("weight", 1.0)),
                        )
            return list(node_map.values()), list(edge_map.values())
        except Exception as e:
            logger.error(f"FalkorDB tag_traversal failed: {e}")
            return [], []

    def get_subgraph_between(
        self, 
        graph_name: str, 
        node_ids: List[str]
    ) -> List[GraphEdge]:
        if not self._connected:
            return self._memory_fallback.get_subgraph_between(graph_name, node_ids)

        if not node_ids:
            return []

        try:
            g = self._get_graph(graph_name)
            query = (
                "MATCH (s)-[r]->(t) "
                "WHERE s.id IN $node_ids AND t.id IN $node_ids "
                "RETURN s.id, type(r), t.id, r"
            )
            res = g.query(query, {"node_ids": node_ids})
            edges = []
            for row in res.result_set:
                s_id, r_type, t_id, r_obj = row
                props = r_obj.properties
                edges.append(
                    GraphEdge(
                        id=props.get("id") or f"{s_id}-{r_type}-{t_id}",
                        source_id=props.get("source_id") or str(s_id),
                        target_id=props.get("target_id") or str(t_id),
                        type=EdgeType(r_type.lower()),
                        weight=float(props.get("weight", 1.0)),
                    )
                )
            return edges
        except Exception as e:
            logger.error(f"FalkorDB get_subgraph_between failed: {e}")
            return []

    def get_nodes_by_source(
        self, 
        graph_name: str, 
        source_uri: str
    ) -> List[GraphNode]:
        if not self._connected:
            return self._memory_fallback.get_nodes_by_source(graph_name, source_uri)

        try:
            g = self._get_graph(graph_name)
            query = "MATCH (n) WHERE n.source_uri = $source_uri RETURN n"
            res = g.query(query, {"source_uri": source_uri})
            nodes = []
            for row in res.result_set:
                nodes.append(GraphNode.from_cypher_props(dict(row[0].properties)))
            return nodes
        except Exception as e:
            logger.error(f"FalkorDB get_nodes_by_source failed: {e}")
            return []

    def delete_nodes(self, graph_name: str, node_ids: List[str]) -> None:
        if not self._connected:
            self._memory_fallback.delete_nodes(graph_name, node_ids)
            return

        if not node_ids:
            return

        try:
            g = self._get_graph(graph_name)
            query = "MATCH (n) WHERE n.id IN $node_ids DETACH DELETE n"
            g.query(query, {"node_ids": node_ids})
        except Exception as e:
            logger.error(f"FalkorDB delete_nodes failed: {e}")

    def get_status(self, graph_name: str) -> Dict[str, Any]:
        if not self._connected:
            status = self._memory_fallback.get_status(graph_name)
            status["backend"] = "memory_fallback (disconnected from FalkorDB)"
            return status

        try:
            g = self._get_graph(graph_name)
            
            n_res = g.query("MATCH (n) RETURN count(n)")
            total_nodes = n_res.result_set[0][0] if n_res.result_set else 0

            e_res = g.query("MATCH ()-[r]->() RETURN count(r)")
            total_edges = e_res.result_set[0][0] if e_res.result_set else 0

            t_res = g.query("MATCH (t:tag) RETURN count(t)")
            total_tags = t_res.result_set[0][0] if t_res.result_set else 0

            type_res = g.query("MATCH (n) RETURN labels(n)[0] as type, count(n)")
            node_types = (
                {str(row[0]).lower(): int(row[1]) for row in type_res.result_set if row[0]}
                if type_res.result_set else {}
            )

            return {
                "backend": "falkordb",
                "graph_name": graph_name,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "total_tags": total_tags,
                "node_types": node_types
            }
        except Exception as e:
            logger.error(f"FalkorDB get_status failed: {e}")
            return {
                "backend": "falkordb (error)",
                "graph_name": graph_name,
                "error": str(e)
            }
