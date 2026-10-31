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

DEFAULT_BASE_URL = "https://api.adaptionlabs.ai"

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
        base_url: str = DEFAULT_BASE_URL,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        assert_build_time()
        self.api_key = api_key or os.getenv("ADAPTIONLABS_API_KEY") or os.getenv(
            "ADAPTION_LABS_API_KEY"
        )
        self.base_url = (base_url or os.getenv("ADAPTIONLABS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, prompt: str) -> Optional[str]:
        assert_build_time()
        if not self.api_key:
            logger.warning("No Adaption Labs API key configured; cannot generate a taxonomy.")
            return None

        payload: Dict[str, object] = {
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        }
        if self.model:
            payload["model"] = self.model

        try:
            response = httpx.post(
                f"{self.base_url}/api/v1/responses",
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
            return self._extract_text(response.json())
        except Exception as exc:
            logger.error("Adaption response was not JSON: %s", exc)
            return None
