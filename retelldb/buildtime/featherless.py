"""
Featherless embeddings, build-time only.

Latent proposals need real vectors. retelldb's runtime embedder degrades to a token
projection when no neural provider is configured, which is the right call for serving and
the wrong one here: token overlap is not latent space, and proposing tag associations from
it would dress a hash collision up as semantics. So this lane refuses to fake it. If the
provider is missing or unhappy, the build fails and says so.
"""

import logging
import os
from typing import List, Optional

import httpx

from retelldb.buildtime.adaption import assert_build_time

logger = logging.getLogger("retelldb.buildtime.featherless")

DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_DIMENSION = 2560
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
API_KEY_ENV = ("FEATHERLESS_API_KEY_2", "FEATHERLESS_API_KEY")

#: known dimensions, so a model swap cannot silently mix vectors
MODEL_DIMENSIONS = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}


class EmbeddingUnavailable(RuntimeError):
    """Raised when a real embedding cannot be computed."""


class FeatherlessEmbedder:
    """
    OpenAI-shaped embeddings client, one call per batch.

    `embed` returns vectors in input order and raises rather than returning placeholders.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ):
        assert_build_time("featherless embeddings")
        self.api_key = api_key or next(
            (os.getenv(name) for name in API_KEY_ENV if os.getenv(name)), None
        )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"featherless:{self.model}"

    def available(self) -> bool:
        return bool(self.api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise EmbeddingUnavailable(
                "no Featherless API key configured (set FEATHERLESS_API_KEY_2 or "
                "FEATHERLESS_API_KEY); latent proposals need a real embedder, so no "
                "token-projection fallback is used here"
            )
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": list(texts)},
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": BROWSER_UA,
                },
                timeout=self.timeout,
            )
        except Exception as exc:
            raise EmbeddingUnavailable(f"embedding request failed: {exc}") from exc

        if response.status_code >= 400:
            raise EmbeddingUnavailable(
                f"embedding endpoint returned {response.status_code}: "
                f"{response.text[:300]}"
            )
        try:
            payload = response.json()
        except Exception as exc:
            raise EmbeddingUnavailable(f"embedding response was not JSON: {exc}") from exc

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingUnavailable(
                f"embedding response had {len(data) if isinstance(data, list) else 'no'} "
                f"items for {len(texts)} inputs"
            )

        vectors: List[List[float]] = []
        for index, item in enumerate(data):
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not vector:
                raise EmbeddingUnavailable(f"embedding {index} came back empty")
            vectors.append([float(value) for value in vector])

        expected = MODEL_DIMENSIONS.get(self.model)
        if expected and len(vectors[0]) != expected:
            raise EmbeddingUnavailable(
                f"{self.model} returned {len(vectors[0])}-dim vectors, expected {expected}"
            )
        return vectors
