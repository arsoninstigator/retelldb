"""
Persona profile schema: the structured description of how a persona sounds and reads.

three layers, kept separate on purpose so a bad interaction is debuggable down to
which layer misfired:

- lingvist: language substrate (languages, code-switching, transliteration, loanwords)
- lexica: the actual words (slang, generational markers, fillers, hedges, idioms)
- lingvica: delivery (register per medium, tone defaults, punctuation-as-tone,
  channel shift corrections)

a persona's profile lives as a node in their own memory graph. memory nodes are
what the persona knows; this profile is how they talk.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class LingvistLayer(BaseModel):
    """Language substrate."""
    primary_languages: List[str] = Field(default_factory=lambda: ["en"])
    code_switch_rate: Optional[float] = None
    code_switch_triggers: List[str] = Field(default_factory=list)
    transliteration_habits: str = ""
    loanword_defaults: List[str] = Field(default_factory=list)


class LexicaLayer(BaseModel):
    """Vocabulary."""
    generational_markers: List[str] = Field(default_factory=list)
    slang_inventory: List[str] = Field(default_factory=list)
    fillers: List[str] = Field(default_factory=list)
    hedges: List[str] = Field(default_factory=list)
    idiom_preferences: List[str] = Field(default_factory=list)


class ToneDefaults(BaseModel):
    """Delivery qualities, each scored 0..1."""
    warmth: float = 0.5
    sarcasm_level: float = 0.0
    playfulness: float = 0.5
    confidence: float = 0.5


class LingvicaLayer(BaseModel):
    """Delivery and register."""
    register_by_medium: Dict[str, str] = Field(default_factory=dict)
    tone_defaults: ToneDefaults = Field(default_factory=ToneDefaults)
    punctuation_as_tone: List[str] = Field(default_factory=list)
    channel_shift_corrections: Dict[str, Any] = Field(default_factory=dict)


class PersonaProfile(BaseModel):
    """A persona's voice profile, stored in the persona's own memory graph."""
    persona_id: str
    display_name: Optional[str] = None
    generated_at: Optional[str] = None
    lingvist: LingvistLayer = Field(default_factory=LingvistLayer)
    lexica: LexicaLayer = Field(default_factory=LexicaLayer)
    lingvica: LingvicaLayer = Field(default_factory=LingvicaLayer)

    def register_for(self, medium: str) -> Optional[str]:
        """The register to use for a given medium, with channel corrections applied by the caller."""
        return self.lingvica.register_by_medium.get(medium)
