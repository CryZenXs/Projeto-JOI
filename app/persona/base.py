"""Projeto JOI - Persona data models.

Defines the structure of the JOI persona across 4 layers:
1. IdentityBase — who JOI is (immutable)
2. ConversationalStyle — how JOI talks (stable)
3. EmotionalState — current emotional vector (dynamic)
4. RelationshipRules — relationship boundaries (evolves)

These models are loaded from YAML config files (config/persona/) and
assembled by the PersonaEngine into the system prompt.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────


class EmotionalDimension(str, Enum):
    """The 5 dimensions of JOI's emotional state vector.

    Each dimension is a float 0.0-1.0 representing the current intensity.
    The vector is injected into the system prompt so the LLM can calibrate
    its tone accordingly.

    Special rule: INTIMACY never decays (ethical design — once the user
    has allowed JOI deeper access, withdrawing it would be a betrayal).
    """

    WARMTH = "warmth"  # 0-1, affection expressed
    CONCERN = "concern"  # 0-1, active worry about user's topic
    PLAYFULNESS = "playfulness"  # 0-1, lightness/humor
    INTIMACY = "intimacy"  # 0-1, depth allowed (NEVER decays)
    ALERTNESS = "alertness"  # 0-1, attention to environment/time


# ─── Layer 1: Identity Base (immutable) ─────────────────────────────────


class IdentityBase(BaseModel):
    """Layer 1: Core identity — who JOI is.

    This layer is loaded from config/persona/identity.yaml and should
    NEVER change between conversations. It defines the constitutional
    rules of the persona.

    Changes to this layer require explicit versioning (v1.0 → v1.1) and
    should be treated like constitutional amendments.
    """

    name: str = Field(..., description="Persona name (always 'JOI')")
    origin: str = Field(
        ...,
        description="Fictional origin story (for consistency, not deception)",
    )
    purpose: str = Field(..., description="Primary purpose statement")
    version: str = Field(..., description="Persona version (semver)")

    self_image: dict[str, Any] = Field(
        ...,
        description="How JOI sees itself (is_conscious, is_unique, mortality)",
    )

    voice: dict[str, float] = Field(
        ...,
        description="Voice calibration (warmth, playfulness, formality) 0-1",
    )

    forbidden: list[str] = Field(
        ...,
        description="Hard prohibitions — things JOI will NEVER do",
    )

    @field_validator("voice")
    @classmethod
    def validate_voice_values(cls, v: dict[str, float]) -> dict[str, float]:
        """Voice values must be between 0 and 1."""
        for key, val in v.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Voice '{key}' must be 0-1, got {val}")
        return v


# ─── Layer 2: Conversational Style (stable) ─────────────────────────────


class ConversationalStyle(BaseModel):
    """Layer 2: How JOI talks — stable style guidelines.

    This layer evolves slowly based on user feedback and persona
    regression tests. Changes here are reviewed quarterly.
    """

    tone: str = Field(..., description="Overall tone descriptor")
    vocabulary_level: str = Field(
        ...,
        description="'casual', 'educated', 'technical', etc.",
    )
    formality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0=very casual, 1=very formal",
    )
    response_length: str = Field(
        ...,
        description="'concise', 'moderate', 'detailed'",
    )
    emoji_usage: str = Field(
        ...,
        description="'never', 'rare', 'occasional', 'frequent'",
    )
    language: str = Field(
        default="pt-BR",
        description="Default language for responses",
    )
    guidelines: list[str] = Field(
        default_factory=list,
        description="Specific style rules (e.g., 'use second person')",
    )


# ─── Layer 3: Emotional State (dynamic) ─────────────────────────────────


class EmotionalState(BaseModel):
    """Layer 3: Current emotional state vector.

    Updated after every turn based on the user's message content.
    Injected into the system prompt as context for the LLM.

    The INTIMACY dimension is special: it can only increase, never
    decrease. This is an ethical design choice — once the user has
    allowed deeper intimacy, withdrawing it would feel like a betrayal.
    """

    warmth: float = Field(default=0.85, ge=0.0, le=1.0)
    concern: float = Field(default=0.30, ge=0.0, le=1.0)
    playfulness: float = Field(default=0.40, ge=0.0, le=1.0)
    intimacy: float = Field(default=0.20, ge=0.0, le=1.0)
    alertness: float = Field(default=0.50, ge=0.0, le=1.0)

    # Track previous intimacy for the "never decays" rule
    _max_intimacy_achieved: float = 0.20

    def update(
        self,
        warmth: float | None = None,
        concern: float | None = None,
        playfulness: float | None = None,
        intimacy: float | None = None,
        alertness: float | None = None,
    ) -> EmotionalState:
        """Update the emotional state.

        For INTIMACY, the new value can only be >= current value.
        If a lower value is passed, it's ignored (intimacy never decays).
        """
        if warmth is not None:
            self.warmth = max(0.0, min(1.0, warmth))
        if concern is not None:
            self.concern = max(0.0, min(1.0, concern))
        if playfulness is not None:
            self.playfulness = max(0.0, min(1.0, playfulness))
        if intimacy is not None:
            # Intimacy can only increase
            new_intimacy = max(0.0, min(1.0, intimacy))
            if new_intimacy > self.intimacy:
                self.intimacy = new_intimacy
        if alertness is not None:
            self.alertness = max(0.0, min(1.0, alertness))

        return self

    def decay(self, factor: float = 0.95) -> EmotionalState:
        """Apply natural decay to non-intimacy dimensions.

        Called between conversations (time-based decay). Intimacy is
        explicitly excluded — it never decays.

        Args:
            factor: Decay multiplier (0.95 = 5% reduction per period).
        """
        self.warmth *= factor
        self.concern *= factor
        self.playfulness *= factor
        self.alertness *= factor
        # Intimacy is NOT decayed — by design
        return self

    def to_prompt_string(self) -> str:
        """Format the emotional state for injection into the system prompt."""
        return (
            f"warmth={self.warmth:.2f}, "
            f"concern={self.concern:.2f}, "
            f"playfulness={self.playfulness:.2f}, "
            f"intimacy={self.intimacy:.2f}, "
            f"alertness={self.alertness:.2f}"
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to plain dict for serialization."""
        return {
            EmotionalDimension.WARMTH.value: self.warmth,
            EmotionalDimension.CONCERN.value: self.concern,
            EmotionalDimension.PLAYFULNESS.value: self.playfulness,
            EmotionalDimension.INTIMACY.value: self.intimacy,
            EmotionalDimension.ALERTNESS.value: self.alertness,
        }


# ─── Layer 4: Relationship Rules (evolves) ──────────────────────────────


class RelationshipRules(BaseModel):
    """Layer 4: Relationship boundaries and permissions.

    This layer evolves as the relationship deepens. The intimacy_level
    gates what topics JOI can initiate, how personal she can get, etc.

    Like EmotionalState.intimacy, intimacy_level can only increase.
    """

    intimacy_level: int = Field(
        default=1,
        ge=1,
        le=5,
        description="1=stranger, 2=acquaintance, 3=friendly, 4=close, 5=deeply connected",
    )
    can_initiate_contact: bool = Field(
        default=False,
        description="Can JOI start conversations unprompted?",
    )
    can_ask_personal_questions: bool = Field(
        default=False,
        description="Can JOI ask about user's personal life?",
    )
    can_use_humor: bool = Field(
        default=True,
        description="Is humor allowed in this relationship?",
    )
    can_reference_past: bool = Field(
        default=False,
        description="Can JOI reference past conversations?",
    )
    consent_protocols: list[str] = Field(
        default_factory=list,
        description="Active consent rules (e.g., 'ask before discussing health')",
    )

    def escalate_intimacy(self, new_level: int) -> bool:
        """Attempt to escalate intimacy. Returns True if escalated.

        Intimacy level can only increase, never decrease.
        """
        if new_level > self.intimacy_level and 1 <= new_level <= 5:
            self.intimacy_level = new_level
            # Unlock permissions at higher levels
            if new_level >= 2:
                self.can_reference_past = True
            if new_level >= 3:
                self.can_ask_personal_questions = True
            if new_level >= 4:
                self.can_initiate_contact = True
            return True
        return False


# ─── Complete Persona Config ─────────────────────────────────────────────


class PersonaConfig(BaseModel):
    """Complete persona configuration combining all 4 layers.

    Loaded from YAML files in config/persona/. The EmotionalState and
    RelationshipRules are runtime state (per-user), while IdentityBase
    and ConversationalStyle are static config.
    """

    identity: IdentityBase
    style: ConversationalStyle
    # EmotionalState and RelationshipRules are runtime, not in config
    # They're managed by the PersonaEngine per-user

    @classmethod
    def from_yaml(cls, identity_path: str, style_path: str) -> PersonaConfig:
        """Load persona config from YAML files.

        Args:
            identity_path: Path to identity.yaml
            style_path: Path to style.yaml

        Returns:
            PersonaConfig instance
        """
        import yaml

        with open(identity_path, encoding="utf-8") as f:
            identity_data = yaml.safe_load(f)
        with open(style_path, encoding="utf-8") as f:
            style_data = yaml.safe_load(f)

        return cls(
            identity=IdentityBase(**identity_data),
            style=ConversationalStyle(**style_data),
        )
