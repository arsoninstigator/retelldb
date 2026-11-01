"""
Persona router.
Resolves which memory graph a query or ingestion targets.

prixie always knows which persona she is running as (the voice loop on this branch
always has the persona id in hand), so the router is deliberate: an explicit persona
routes to that persona's sandboxed memory. no hint means shared memory, the stuff
every persona knows. there is no guessing "developer or user" anymore: personas are
arbitrary (hackathon you, class you, work you), and misfiling a memory into the
wrong persona's head is worse than landing it in shared.
"""

from typing import Optional, Tuple
from retelldb.models.schema import SHARED_NAMESPACE, normalize_namespace


class PersonaRouter:
    """Resolves a memory namespace (and its graph) from a persona hint."""

    def __init__(self, config=None):
        self.config = config

    def resolve(
        self,
        persona_id: Optional[str] = None,
        namespace: Optional[str] = None,
        persona_hint: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Return (namespace, target_graph) for this query or ingest.

        precedence: raw namespace override > persona_id > shared.
        """
        if namespace:
            ns = normalize_namespace(namespace)
        elif persona_id or persona_hint:
            ns = normalize_namespace(persona_id or persona_hint)
        else:
            ns = SHARED_NAMESPACE

        if self.config is not None:
            return ns, self.config.storage.graph_for(ns)
        return ns, ns
