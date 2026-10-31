"""
Adaption Labs transport, build-time only.

This provider generates config during a build. It must never be reachable from a
production deployment, so every entry point calls `assert_build_time` first and raises
rather than degrading into a network call nobody expected.
"""

import logging
import os
from typing import Dict, Optional

import httpx

from retelldb.wrangling.generator import DEFAULT_TIMEOUT, extract_text

logger = logging.getLogger("retelldb.buildtime.adaption")

DEFAULT_BASE_URL = "https://api.prod.adaptionlabs.ai"
#: every call must name a model; the endpoint rejects a request without one
#: measured on the retelldb eval corpus behind a gateway that closes at 60s: gemma-4-31b-it
#: answers a 12-tag batch in ~22s with the cleanest hierarchy (1 dangling reference, 3
#: uncovered tags). deepseek-v4-flash takes ~35s and drifts, glm-5.2 and qwen3.8-27b spend the
#: whole window reasoning and get 504'd, so both are opt-in via --model.
DEFAULT_MODEL = "gemma-4-31b-it"
#: the routes live under /api/v1 on this host, even though the docs call it /v1
API_PREFIX = "/api/v1"


def normalize_base_url(base_url: Optional[str]) -> str:
    """
    Accept the host with or without the version suffix.

    `https://api.prod.adaptionlabs.ai`, `.../v1` and `.../api/v1` all mean the same base,
    because the transport appends `/api/v1/responses` itself.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_BASE_URL
    for suffix in ("/api/v1", "/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/") or DEFAULT_BASE_URL

#: deployment signals that mean "not a build"
PRODUCTION_VALUES = {"production", "prod", "live"}
DEPLOYMENT_ENV_VARS = (
    "RETELLDB_DEPLOYMENT",
    "RETELLDB_ENV",
    "DEPLOYMENT_ENV",
    "ENVIRONMENT",
)


def assert_build_time(context: str = "adaption") -> None:
    """
    Refuse to talk to a build-time provider from a production deployment.

    Called by every entry point instead of documenting the rule and hoping. A missing
    signal counts as a build, because that is a local run or a CI job.
    """
    for name in DEPLOYMENT_ENV_VARS:
        value = (os.getenv(name) or "").strip().lower()
        if value in PRODUCTION_VALUES:
            raise RuntimeError(
                f"{context} is build-time only and refuses to run with {name}={value}. "
                "generate taxonomies during a build, never in a production deployment."
            )


class AdaptionTransport:
    """
    Minimal client for Adaption's hosted model-response endpoint.

    The transport's only job is text in, text out. Any HTTP or shape surprise becomes
    a logged None so the generator can report failure instead of guessing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        assert_build_time()
        self.api_key = api_key or os.getenv("ADAPTIONLABS_API_KEY") or os.getenv(
            "ADAPTION_LABS_API_KEY"
        )
        self.base_url = normalize_base_url(
            base_url or os.getenv("ADAPTIONLABS_BASE_URL")
        )
        self.model = model or os.getenv("ADAPTIONLABS_MODEL") or DEFAULT_MODEL
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, prompt: str) -> Optional[str]:
        assert_build_time()
        if not self.api_key:
            logger.warning("No Adaption Labs API key configured; cannot generate a taxonomy.")
            return None

        payload: Dict[str, object] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = httpx.post(
                f"{self.base_url}{API_PREFIX}/responses",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.error("Adaption request failed: %s", exc)
            return None

        if response.status_code >= 400:
            logger.error(
                "Adaption response error %s: %s",
                response.status_code,
                response.text[:300],
            )
            return None

        try:
            return extract_text(response.json())
        except Exception as exc:
            logger.error("Adaption response was not JSON: %s", exc)
            return None
