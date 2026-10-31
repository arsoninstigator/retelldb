"""
Canonical tag ontology and alias resolution.

The rule this module enforces: an alias is provenance, not a cleanup side effect.
A raw tag is never rewritten into its canonical form. The raw string survives on
the node, and the resolution is written to the graph as an `alias_of` edge from
an alias tag node to its canonical tag node, weighted by resolution confidence.

Ambiguity is never guessed. If an alias maps to two canonical tags it resolves to
nothing, and the caller keeps the raw tag as its own canonical.
"""

import json
import logging
import pathlib
import re
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger("retelldb.wrangling.taxonomy")

_SEPARATOR_RE = re.compile(r"[\s_\-/]+")


def normalize_tag(raw: str) -> str:
    """Normalize a tag for comparison only. The raw form is always preserved."""
    return _SEPARATOR_RE.sub(" ", str(raw).strip().lower()).strip()


class ResolutionRule(str, Enum):
    """How a raw tag reached its canonical form."""

    EXACT = "exact"          # raw string is already a canonical tag
    ALIAS = "alias"          # raw string matched a configured alias
    AMBIGUOUS = "ambiguous"  # raw string matches more than one canonical tag
    UNRESOLVED = "unresolved"  # nothing matched, raw tag stays its own canonical


class AliasRule(BaseModel):
    """One configured alias pointing at a canonical tag."""

    alias: str
    confidence: float = 0.95


class CanonicalTag(BaseModel):
    """A canonical tag with its aliases, hierarchy and associations."""

    name: str
    aliases: List[AliasRule] = Field(default_factory=list)
    # parent tags, weight 1.0: a child tag implies its parent (python -> programming)
    parents: List[str] = Field(default_factory=list)
    # explicit association strengths, never inferred from vector distance
    related: Dict[str, float] = Field(default_factory=dict)
    # mutually exclusive tags: an edge between these is never proposed or stored
    excludes: List[str] = Field(default_factory=list)

    def normalized_name(self) -> str:
        return normalize_tag(self.name)


class TagResolution(BaseModel):
    """The auditable result of resolving one raw tag."""

    raw: str
    canonical: Optional[str] = None
    confidence: float = 0.0
    rule: ResolutionRule = ResolutionRule.UNRESOLVED
    candidates: List[str] = Field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """True when a canonical form was actually identified."""
        return self.canonical is not None and self.rule in (
            ResolutionRule.EXACT,
            ResolutionRule.ALIAS,
        )

    @property
    def is_alias(self) -> bool:
        """True when the raw tag differs from its canonical form."""
        return self.rule == ResolutionRule.ALIAS


class TagTaxonomy(BaseModel):
    """
    The configured tag ontology for one project.

    An empty taxonomy is valid and means: every raw tag is its own canonical tag,
    with no aliasing and no hierarchy. That is the honest default. Canonicalisation
    is configuration, not something the engine invents at query time.
    """

    tags: List[CanonicalTag] = Field(default_factory=list)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TagTaxonomy":
        if not data:
            return cls()
        raw_tags = data.get("tags", [])
        tags: List[CanonicalTag] = []
        for entry in raw_tags:
            if isinstance(entry, str):
                tags.append(CanonicalTag(name=entry))
                continue
            aliases = []
            for alias in entry.get("aliases", []) or []:
                if isinstance(alias, str):
                    aliases.append(AliasRule(alias=alias))
                else:
                    aliases.append(
                        AliasRule(
                            alias=str(alias["alias"]),
                            confidence=float(alias.get("confidence", 0.95)),
                        )
                    )
            related = {
                normalize_tag(k): float(v)
                for k, v in (entry.get("related", {}) or {}).items()
            }
            tags.append(
                CanonicalTag(
                    name=normalize_tag(entry["name"]),
                    aliases=aliases,
                    parents=[normalize_tag(p) for p in entry.get("parents", []) or []],
                    related=related,
                    excludes=[
                        normalize_tag(x) for x in entry.get("excludes", []) or []
                    ],
                )
            )
        return cls(tags=tags)

    @classmethod
    def from_file(cls, path: Union[str, pathlib.Path]) -> "TagTaxonomy":
        """Load a taxonomy from a JSON or YAML file."""
        text = pathlib.Path(path).read_text(encoding="utf-8")
        if str(path).lower().endswith((".yaml", ".yml")):
            try:
                import yaml  # already a hard dependency of the package
            except ImportError:  # pragma: no cover - pyyaml is a declared dependency
                raise RuntimeError("pyyaml is required to load YAML taxonomies")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    # -- resolution -------------------------------------------------------

    def canonical_names(self) -> List[str]:
        return [t.normalized_name() for t in self.tags]

    def alias_index(self) -> Dict[str, List[AliasRule]]:
        """Map normalized alias -> list of alias rules (one per canonical tag)."""
        index: Dict[str, List[AliasRule]] = {}
        for tag in self.tags:
            for rule in tag.aliases:
                index.setdefault(normalize_tag(rule.alias), []).append(rule)
        return index

    def parents_of(self, canonical: str) -> List[str]:
        target = normalize_tag(canonical)
        for tag in self.tags:
            if tag.normalized_name() == target:
                return list(tag.parents)
        return []

    def resolve(self, raw: str) -> TagResolution:
        """Resolve one raw tag. An ambiguous alias resolves to nothing."""
        normalized = normalize_tag(raw)
        if not normalized:
            return TagResolution(raw=str(raw), rule=ResolutionRule.UNRESOLVED)

        canonicals = set(self.canonical_names())
        if normalized in canonicals:
            return TagResolution(
                raw=str(raw),
                canonical=normalized,
                confidence=1.0,
                rule=ResolutionRule.EXACT,
            )

        return self._resolve_alias(raw, normalized)

    def _resolve_alias(self, raw: str, normalized: str) -> TagResolution:
        # AliasRule does not carry its owning canonical tag, so walk the tags.
        owners: List[tuple] = []
        for tag in self.tags:
            for rule in tag.aliases:
                if normalize_tag(rule.alias) == normalized:
                    owners.append((tag.normalized_name(), rule.confidence))

        distinct = sorted({owner for owner, _ in owners})
        if len(distinct) > 1:
            logger.warning(
                "Ambiguous tag alias '%s' maps to %s; keeping the raw tag as its own canonical.",
                raw,
                distinct,
            )
            return TagResolution(
                raw=str(raw),
                canonical=None,
                confidence=0.0,
                rule=ResolutionRule.AMBIGUOUS,
                candidates=distinct,
            )

        if len(distinct) == 1:
            canonical, confidence = owners[0]
            return TagResolution(
                raw=str(raw),
                canonical=canonical,
                confidence=confidence,
                rule=ResolutionRule.ALIAS if canonical != normalized else ResolutionRule.EXACT,
            )

        # nothing configured: the raw tag is its own canonical, at low confidence
        return TagResolution(
            raw=str(raw),
            canonical=normalized,
            confidence=0.5,
            rule=ResolutionRule.UNRESOLVED,
        )

    def resolve_all(self, raw_tags: List[str]) -> List[TagResolution]:
        return [self.resolve(tag) for tag in raw_tags]


def taxonomy_from_config(config) -> TagTaxonomy:
    """
    Load the tag taxonomy a config points at, or return an empty one.

    A taxonomy that fails to load must never silently start aliasing: the loader
    logs and falls back to no aliasing at all.
    """
    path = getattr(config, "taxonomy_path", None)
    if not path:
        wrangling = getattr(config, "wrangling", None)
        path = getattr(wrangling, "taxonomy_path", None)
    if not path:
        return TagTaxonomy()
    try:
        return TagTaxonomy.from_file(path)
    except Exception as exc:
        logger.error("Failed to load tag taxonomy from %s: %s", path, exc)
        return TagTaxonomy()
