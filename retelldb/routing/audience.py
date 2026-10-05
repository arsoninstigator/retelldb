"""
Audience classifier.
Routes incoming queries to either the developer docs graph or the user docs graph.
"""

from typing import Optional
import re
from retelldb.models.schema import AudienceType
from retelldb.config import InferenceConfig, default_config


class AudienceClassifier:
    """Classifies whether a query belongs to the developer graph or user graph."""

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or default_config.inference
        self._litellm_available = False
        try:
            import litellm
            self.litellm = litellm
            self._litellm_available = True
        except ImportError:
            self.litellm = None

    def classify(self, query: str, audience_hint: Optional[AudienceType] = None) -> AudienceType:
        """Return the target audience, using hint if provided or predicting via fast model / heuristics."""
        if audience_hint is not None:
            return audience_hint

        # Try fast model tier if available
        if self._litellm_available and self.config.api_key:
            try:
                prompt = (
                    f"Classify the following query target audience as either 'developer' or 'user'.\n"
                    f"Query: {query}\n"
                    f"Respond only with one word: 'developer' or 'user'."
                )
                resp = self.litellm.completion(
                    model=self.config.fast_model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=self.config.api_key,
                    max_tokens=10,
                    temperature=0.0
                )
                pred = resp.choices[0].message.content.strip().lower()
                if "user" in pred:
                    return AudienceType.USER
                return AudienceType.DEVELOPER
            except Exception:
                pass

        # Fast heuristic classification
        q_lower = query.lower()
        dev_signals = [
            "api", "sdk", "endpoint", "curl", "python", "javascript", "typescript",
            "token", "bearer", "param", "header", "payload", "json", "rest", "graphql",
            "error code", "status 40", "status 50", "exception", "async", "function",
            "method", "class", "import", "npm", "pip", "webhook", "jwt", "auth header"
        ]
        user_signals = [
            "how to use", "dashboard", "billing", "pricing", "invoice", "credit card",
            "account settings", "login screen", "forgot password", "invite team", "workspace",
            "export pdf", "download csv", "tutorial for beginners", "subscription"
        ]

        dev_score = sum(1 for s in dev_signals if re.search(rf"\b{re.escape(s)}\b", q_lower))
        user_score = sum(1 for s in user_signals if re.search(rf"\b{re.escape(s)}\b", q_lower))

        if user_score > dev_score:
            return AudienceType.USER
        return AudienceType.DEVELOPER
