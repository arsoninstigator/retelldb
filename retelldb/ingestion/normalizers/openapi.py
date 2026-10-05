"""
OpenAPI and AsyncAPI specification normalizer.
Transforms endpoints, parameters, schemas, and error responses into typed nodes
and explicit relationship edges (depends_on, implements, example_of) with deterministic IDs.
"""

import json
import yaml
from typing import Optional, Dict, Any, List
from retelldb.ingestion.normalizers.base import BaseNormalizer, NormalizedDocument, CandidateNode, CandidateEdge
from retelldb.models.schema import NodeType, EdgeType, AudienceType


class OpenAPINormalizer(BaseNormalizer):
    """Normalizes OpenAPI 3.x and AsyncAPI 2.x/3.x specs into documentation graph nodes."""

    def normalize(
        self,
        content: str,
        audience: AudienceType = AudienceType.DEVELOPER,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        **kwargs
    ) -> NormalizedDocument:
        spec = self._parse_spec(content)
        
        info = spec.get("info", {})
        version = info.get("version", "1.0.0")
        doc_title = title or f"{info.get('title', 'API Specification')} (v{version})"

        nodes: List[CandidateNode] = []
        edges: List[CandidateEdge] = []
        global_tags: List[str] = ["api", "openapi"]

        # Root API doc node
        root_node = CandidateNode.create_deterministic(
            title=doc_title,
            type=NodeType.GUIDE,
            content=info.get("description", f"API specification for {doc_title}"),
            audience=audience,
            tags=global_tags.copy(),
            source_uri=source_uri,
            doc_title=doc_title,
            locator="root",
            metadata={"version": version, "source_uri": source_uri, "is_root": True}
        )
        nodes.append(root_node)

        # Parse Schemas / Models
        schema_nodes: Dict[str, CandidateNode] = {}
        schemas = spec.get("components", {}).get("schemas", {})
        for schema_name, schema_data in schemas.items():
            locator = f"schema:{schema_name}"
            schema_node = CandidateNode.create_deterministic(
                title=f"Schema: {schema_name}",
                type=NodeType.CONCEPT,
                content=json.dumps(schema_data, indent=2),
                audience=audience,
                tags=list(set(global_tags + [schema_name.lower(), "schema", "model"])),
                source_uri=source_uri,
                doc_title=doc_title,
                locator=locator,
                metadata={"schema_name": schema_name, "type": schema_data.get("type", "object")}
            )
            nodes.append(schema_node)
            schema_nodes[schema_name] = schema_node
            
            # Connect schema to root
            edges.append(
                CandidateEdge.create_deterministic(
                    source_id=schema_node.id,
                    target_id=root_node.id,
                    type=EdgeType.DEPENDS_ON,
                    weight=0.7
                )
            )

        # Parse Endpoints (paths)
        paths = spec.get("paths", {})
        for path, path_item in paths.items():
            for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
                if method not in path_item:
                    continue
                
                op = path_item[method]
                op_id = op.get("operationId", f"{method.upper()} {path}")
                summary = op.get("summary") or op_id
                description = op.get("description", "")
                ep_tags = [t.lower() for t in op.get("tags", [])]
                ep_locator = f"endpoint:{method.upper()}:{path}"

                endpoint_node = CandidateNode.create_deterministic(
                    title=f"{method.upper()} {path}",
                    type=NodeType.API_ENDPOINT,
                    content=f"Summary: {summary}\nDescription: {description}\nOperation: {op_id}",
                    audience=audience,
                    tags=list(set(global_tags + ep_tags + [method.lower()])),
                    source_uri=source_uri,
                    doc_title=doc_title,
                    locator=ep_locator,
                    metadata={
                        "path": path,
                        "method": method.upper(),
                        "operation_id": op_id,
                        "parameters": op.get("parameters", []),
                    }
                )
                nodes.append(endpoint_node)

                # Connect endpoint to root
                edges.append(
                    CandidateEdge.create_deterministic(
                        source_id=endpoint_node.id,
                        target_id=root_node.id,
                        type=EdgeType.IMPLEMENTS,
                        weight=0.9
                    )
                )

                # Connect referenced schemas
                content_str = json.dumps(op)
                for s_name, s_node in schema_nodes.items():
                    if f"#/components/schemas/{s_name}" in content_str or s_name in content_str:
                        edges.append(
                            CandidateEdge.create_deterministic(
                                source_id=endpoint_node.id,
                                target_id=s_node.id,
                                type=EdgeType.DEPENDS_ON,
                                weight=0.9
                            )
                        )

                # Check error responses
                responses = op.get("responses", {})
                for status_code, resp in responses.items():
                    if status_code.startswith("4") or status_code.startswith("5"):
                        err_locator = f"{ep_locator}:error:{status_code}"
                        error_node = CandidateNode.create_deterministic(
                            title=f"Error {status_code} on {method.upper()} {path}",
                            type=NodeType.ERROR,
                            content=resp.get("description", f"Error response {status_code}"),
                            audience=audience,
                            tags=list(set(ep_tags + ["error", f"status-{status_code}"])),
                            source_uri=source_uri,
                            doc_title=doc_title,
                            locator=err_locator,
                            metadata={"status_code": status_code, "endpoint": f"{method.upper()} {path}"}
                        )
                        nodes.append(error_node)
                        edges.append(
                            CandidateEdge.create_deterministic(
                                source_id=error_node.id,
                                target_id=endpoint_node.id,
                                type=EdgeType.RELATED,
                                weight=0.8
                            )
                        )

        return NormalizedDocument(
            document_title=doc_title,
            source_uri=source_uri,
            audience=audience,
            nodes=nodes,
            edges=edges,
            global_tags=global_tags
        )

    def _parse_spec(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except Exception:
            return yaml.safe_load(content) or {}
