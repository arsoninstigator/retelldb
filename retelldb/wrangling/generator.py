"""
Taxonomy generation.

The wrangler resolves tags against a configured ontology, and this module builds that
ontology: it reads the raw tags a project actually produced, asks a generator model to
propose canonical tags, aliases and hierarchy, then validates the answer before it is
allowed to become configuration.

Two rules keep this honest:

1. Nothing is invented. Every observed raw tag ends up either as a canonical tag, as
   an alias of one, or explicitly reported as unclassified. A generated alias is a
   suggestion written into a file a human reviews, never a silent graph mutation.
2. A failed generation fails. If the provider is unreachable or returns something
   that is not a valid taxonomy, the caller gets an error, not a half-empty ontology.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import httpx
from pydantic import BaseModel, Field

from retelldb.wrangling.taxonomy import AliasRule, CanonicalTag, TagTaxonomy, normalize_tag

logger = logging.getLogger("retelldb.wrangling.generator")

DEFAULT_TIMEOUT = 60.0
# the hosted endpoint sits behind a gateway that closes the connection at 60s, and a
# reasoning model finishes roughly 12 tags inside that window. 16 took 57s, 20 timed out.
DEFAULT_BATCH_SIZE = 12
MAX_SNIPPET_CHARS = 400


class GeneratorTransport(Protocol):
    """Text in, text out. Every build-time provider satisfies this and nothing else."""

    def available(self) -> bool: ...

    def complete(self, system: str, prompt: str) -> Optional[str]: ...


class GenerationError(RuntimeError):
    """Raised when a taxonomy could not be produced honestly."""


class TagInventory(BaseModel):
    """The raw material: tags a project actually emitted, plus where they came from."""

    tags: List[str] = Field(default_factory=list)
    snippets: Dict[str, str] = Field(default_factory=dict)

    def batches(self, size: int = DEFAULT_BATCH_SIZE) -> List[List[str]]:
        return [self.tags[i:i + size] for i in range(0, len(self.tags), size)]


def collect_inventory(
    tags: Sequence[str], snippets: Optional[Dict[str, str]] = None
) -> TagInventory:
    """Normalize and deduplicate an observed tag set."""
    seen: Dict[str, None] = {}
    for tag in tags:
        clean = normalize_tag(tag)
        if clean:
            seen.setdefault(clean, None)
    return TagInventory(
        tags=sorted(seen.keys()),
        snippets={k.lower(): v for k, v in (snippets or {}).items()},
    )


@staticmethod
def extract_text(data: object) -> Optional[str]:
    """Pull text out of the response shapes the endpoint may use."""
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return None

    for key in ("output_text", "text", "content"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value

    output = data.get("output")
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            # reasoning items carry their chain of thought in the same shape as answers,
            # and feeding that to the parser breaks the json contract
            if item.get("type") not in (None, "message"):
                continue
            for entry in item.get("content") or []:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") not in (None, "output_text"):
                    continue
                if isinstance(entry.get("text"), str):
                    parts.append(entry["text"])
        if parts:
            return "\n".join(parts)

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

    return None


class OpenAICompatTransport:
    """
    Chat-completions transport for any OpenAI-shaped endpoint.

    Kept separate from the Adaption transport because the two have different auth and
    payload shapes. Used as an alternate rail when the primary provider is unavailable;
    a dead provider should not block generation when another one is configured.
    """

    BROWSER_UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.featherless.ai/v1",
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        timeout: float = DEFAULT_TIMEOUT,
        api_key_env: Sequence[str] = ("FEATHERLESS_API_KEY_2", "FEATHERLESS_API_KEY"),
    ):
        import os

        self.api_key = api_key or next(
            (os.getenv(name) for name in api_key_env if os.getenv(name)), None
        )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, prompt: str) -> Optional[str]:
        if not self.api_key:
            logger.warning("No API key for the OpenAI-compatible transport.")
            return None
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": self.BROWSER_UA,
                },
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.error("Chat completion request failed: %s", exc)
            return None

        if response.status_code >= 400:
            logger.error(
                "Chat completion error %s: %s",
                response.status_code,
                response.text[:300],
            )
            return None
        try:
            return extract_text(response.json())
        except Exception as exc:
            logger.error("Chat completion response was not JSON: %s", exc)
            return None


SYSTEM_PROMPT = (
    "You maintain a canonical tag ontology for a documentation retrieval graph. "
    "You receive raw tags that humans actually typed. You return strict JSON only, "
    "with no prose and no markdown fences."
)


def build_prompt(tags: Sequence[str], snippets: Optional[Dict[str, str]] = None) -> str:
    """Ask for canonical tags, aliases with confidence, hierarchy, and ambiguity."""
    lines = [
        "Raw tags observed in the corpus:",
        json.dumps(list(tags), ensure_ascii=False),
        "",
    ]
    if snippets:
        lines.append("Sample context for some tags (title: snippet):")
        for tag in tags:
            snippet = (snippets or {}).get(tag)
            if snippet:
                lines.append(f"- {tag}: {snippet[:MAX_SNIPPET_CHARS]}")
        lines.append("")

    lines.extend(
        [
            "Produce a JSON object with two keys.",
            '1. "tags": a list of canonical tags. Each entry is',
            '   {"name": str, "aliases": [{"alias": str, "confidence": 0-1}],',
            '    "parents": [str], "related": {str: 0-1}}.',
            "   Rules:",
            "   - every raw tag given above must appear either as a canonical name or as",
            "     an alias of one; do not drop any",
            "   - keep canonical names lowercase, short and stable",
            "   - parents are broader tags; only add a parent when it is genuinely broader",
            "   - alias confidence reflects how sure you are that the alias means that",
            "     canonical tag, and never exceeds 0.99",
            '2. "ambiguous": a list of raw tags whose meaning depends on context, with',
            '   the competing canonical tags, as',
            '   {"raw": str, "candidates": [str], "note": str}.',
            "",
            "Return only the JSON object.",
        ]
    )
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    return fence.group(1) if fence else stripped


def parse_taxonomy(text: str) -> Tuple[TagTaxonomy, List[Dict[str, object]]]:
    """Parse and validate a generated taxonomy. Invalid output is an error, not a guess."""
    try:
        data = json.loads(_strip_fences(text))
    except (ValueError, TypeError) as exc:
        raise GenerationError(f"generator did not return JSON: {exc}") from exc

    if not isinstance(data, dict) or "tags" not in data:
        raise GenerationError("generator JSON is missing the 'tags' key")

    raw_tags = data.get("tags")
    if not isinstance(raw_tags, list) or not raw_tags:
        raise GenerationError("generator returned no tags")

    parsed: List[CanonicalTag] = []
    for entry in raw_tags:
        if isinstance(entry, str):
            parsed.append(CanonicalTag(name=normalize_tag(entry)))
            continue
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        aliases: List[AliasRule] = []
        for alias in entry.get("aliases", []) or []:
            if isinstance(alias, str):
                aliases.append(AliasRule(alias=normalize_tag(alias), confidence=0.95))
            elif isinstance(alias, dict) and alias.get("alias"):
                try:
                    confidence = float(alias.get("confidence", 0.95))
                except (TypeError, ValueError):
                    confidence = 0.95
                aliases.append(
                    AliasRule(
                        alias=normalize_tag(alias["alias"]),
                        confidence=max(0.05, min(0.99, confidence)),
                    )
                )
        related: Dict[str, float] = {}
        for key, value in (entry.get("related") or {}).items():
            try:
                related[normalize_tag(key)] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        parsed.append(
            CanonicalTag(
                name=normalize_tag(entry["name"]),
                aliases=aliases,
                parents=[normalize_tag(p) for p in (entry.get("parents") or []) if p],
                related=related,
            )
        )

    if not parsed:
        raise GenerationError("generator JSON contained no usable canonical tags")

    ambiguous = [
        item
        for item in (data.get("ambiguous") or [])
        if isinstance(item, dict) and item.get("raw")
    ]
    return TagTaxonomy(tags=parsed), ambiguous


def reconcile(taxonomy: TagTaxonomy, inventory: TagInventory) -> Dict[str, object]:
    """
    Check the generated taxonomy against what was actually observed.

    Returns a report: tags left uncovered, aliases pointing at tags that do not exist,
    and aliases that collide with a canonical name.
    """
    canonicals = set(taxonomy.canonical_names())
    covered = set(canonicals)
    alias_targets: List[str] = []
    collisions: List[str] = []

    for tag in taxonomy.tags:
        for rule in tag.aliases:
            alias = normalize_tag(rule.alias)
            covered.add(alias)
            alias_targets.append(alias)
            if alias in canonicals and alias != tag.normalized_name():
                collisions.append(alias)

    uncovered = sorted(t for t in inventory.tags if t not in covered)
    dangling = sorted(
        {p for tag in taxonomy.tags for p in tag.parents if p not in canonicals}
        | {r for tag in taxonomy.tags for r in tag.related if r not in canonicals}
    )

    return {
        "canonical_count": len(canonicals),
        "alias_count": len(alias_targets),
        "uncovered_raw_tags": uncovered,
        "dangling_references": dangling,
        "alias_collisions": sorted(set(collisions)),
    }


def generate_taxonomy(
    inventory: TagInventory,
    transport: "GeneratorTransport",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[TagTaxonomy, Dict[str, object]]:
    """
    Generate a taxonomy for an observed tag inventory.

    Batches are merged; a batch that fails fails the whole generation, because a
    partial ontology would silently leave parts of the corpus unresolved.
    """
    if not inventory.tags:
        raise GenerationError("no tags to generate from")
    if not transport.available():
        raise GenerationError(
            "the configured generator transport has no credentials "
            f"({type(transport).__name__})"
        )

    merged: Dict[str, CanonicalTag] = {}
    ambiguities: List[Dict[str, object]] = []

    for batch in inventory.batches(batch_size):
        text = transport.complete(SYSTEM_PROMPT, build_prompt(batch, inventory.snippets))
        if not text:
            raise GenerationError(
                f"generator returned nothing for a batch of {len(batch)} tags"
            )
        batch_taxonomy, batch_ambiguous = parse_taxonomy(text)
        for tag in batch_taxonomy.tags:
            existing = merged.get(tag.normalized_name())
            if existing is None:
                merged[tag.normalized_name()] = tag
                continue
            # merge aliases and hierarchy from the second mention
            known_aliases = {normalize_tag(a.alias) for a in existing.aliases}
            for rule in tag.aliases:
                if normalize_tag(rule.alias) not in known_aliases:
                    existing.aliases.append(rule)
            for parent in tag.parents:
                if parent not in existing.parents:
                    existing.parents.append(parent)
            for related, weight in tag.related.items():
                existing.related.setdefault(related, weight)
        ambiguities.extend(batch_ambiguous)

    taxonomy = TagTaxonomy(tags=sorted(merged.values(), key=lambda t: t.name))
    report = reconcile(taxonomy, inventory)
    report["ambiguous"] = ambiguities
    return taxonomy, report


def taxonomy_to_yaml(taxonomy: TagTaxonomy) -> str:
    """Serialize a taxonomy to the reviewable YAML config format."""
    import yaml

    payload = {
        "tags": [
            {
                "name": tag.name,
                "aliases": [
                    {"alias": rule.alias, "confidence": round(rule.confidence, 3)}
                    for rule in tag.aliases
                ],
                **({"parents": list(tag.parents)} if tag.parents else {}),
                **({"related": dict(tag.related)} if tag.related else {}),
            }
            for tag in taxonomy.tags
        ]
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
