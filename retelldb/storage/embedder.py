"""
Context-aware node embedding generator.
Appends neighboring node titles, tags, and type metadata to node content
before computing embeddings to enable associative proximity search.
Includes capability verification and explicit warnings for degraded fallback modes.
"""

from typing import List, Optional, Dict, Any
import math
import hashlib
import re
import logging
from retelldb.models.schema import GraphNode
from retelldb.config import InferenceConfig, default_config

logger = logging.getLogger("retelldb.embedder")


class ContextualEmbedder:
    """Generates context-rich embeddings for documentation nodes and queries."""

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or default_config.inference
        self._litellm_available = False
        self.is_neural = False
        self.provider_info = "local_token_projection"
        self.last_warning = None
        self._checked = False

        try:
            import litellm
            self.litellm = litellm
            self._litellm_available = True
        except ImportError:
            self.litellm = None

        self._check_capabilities()

    def _check_capabilities(self) -> None:
        """Startup capability check to ensure embedding provider is verified."""
        if self._checked:
            return
        self._checked = True

        api_key = self.config.embedding_api_key or self.config.api_key
        if not self._litellm_available:
            self.is_neural = False
            self.provider_info = "local_token_projection (litellm not installed)"
            logger.warning(
                "⚠️  EMBEDDING DEGRADATION WARNING: 'litellm' is not installed. "
                "RetellDB is operating in deterministic token projection fallback mode. "
                "Vector similarity uses word token hashing rather than dense neural associative embeddings."
            )
            return

        if not api_key:
            self.is_neural = False
            self.provider_info = "local_token_projection (no API key configured)"
            logger.warning(
                "⚠️  EMBEDDING DEGRADATION WARNING: No API key provided for embedding model. "
                f"Configured model: '{self.config.embedding_model}'. "
                "RetellDB is operating in deterministic token projection fallback mode. "
                "To enable dense neural associative retrieval, set OPENAI_API_KEY or FEATHERLESS_API_KEY."
            )
            return

        # Attempt test probe
        try:
            resp = self.litellm.embedding(
                model=self.config.embedding_model,
                input=["retelldb probe"],
                api_key=api_key
            )
            if resp and resp.data:
                self.is_neural = True
                self.provider_info = f"neural:{self.config.embedding_model}"
                logger.info(f"Verified neural embedding provider: {self.config.embedding_model}")
        except Exception as e:
            self.is_neural = False
            self.provider_info = f"local_token_projection (probe failed: {type(e).__name__})"
            self.last_warning = str(e)
            logger.warning(
                f"⚠️  EMBEDDING DEGRADATION WARNING: LiteLLM probe failed for model '{self.config.embedding_model}': {e}. "
                "RetellDB has fallen back to deterministic token projection. "
                "Associative similarity will run via token overlap rather than latent neural embeddings."
            )

    def build_contextual_text(
        self, 
        node: GraphNode, 
        neighbor_titles: Optional[List[str]] = None
    ) -> str:
        """Constructs associative context: node content + tag labels + neighboring node titles."""
        parts = [
            f"Title: {node.title}",
            f"Type: {node.type.value}",
        ]
        if node.tags:
            parts.append(f"Tags: {', '.join(node.tags)}")
        if neighbor_titles:
            parts.append(f"Connected Context: {', '.join(neighbor_titles)}")
        parts.append(f"Content: {node.content}")
        return "\n".join(parts)

    def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for text."""
        api_key = self.config.embedding_api_key or self.config.api_key
        if self._litellm_available and api_key and self.is_neural:
            try:
                response = self.litellm.embedding(
                    model=self.config.embedding_model,
                    input=[text],
                    api_key=api_key,
                )
                return response.data[0]["embedding"]
            except Exception as e:
                logger.error(
                    f"Neural embedding call failed unexpectedly ({e}). "
                    "Temporarily utilizing deterministic token projection for this item."
                )

        # Deterministic token projection (384-dim) for zero-dependency operation
        return self._pseudo_semantic_embedding(text, dim=384)

    def get_status(self) -> Dict[str, Any]:
        """Observability status for embedding subsystem."""
        return {
            "is_neural": self.is_neural,
            "provider_info": self.provider_info,
            "model": self.config.embedding_model if self.is_neural else None,
            "fallback_active": not self.is_neural,
            "last_warning": self.last_warning
        }

    def _pseudo_semantic_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        Deterministic word-token projection fallback for tests and offline usage.
        Encodes token frequencies into high-dimensional space so semantic overlap
        yields high cosine similarity.
        """
        vec = [0.0] * dim
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) & 1) == 1 else -1.0
            weight = math.log(1.0 + len(token))
            vec[idx] += sign * weight

        # Normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
