"""Projeto JOI - Persona module.

Implements the JOI persona system with 4-layer system prompt architecture:

    Layer 1: Identity Base (immutable)
        - Name, origin, purpose, self-image, forbidden behaviors
        - Fixed at v1.0, never changes between conversations

    Layer 2: Conversational Style (stable, quarterly review)
        - Tone, vocabulary, formality, emoji usage, response length
        - Evolves slowly based on user feedback

    Layer 3: Emotional Memory (dynamic, per-turn)
        - Current emotional state vector (warmth, concern, playfulness, etc.)
        - Recent emotional context from memory system
        - Updates every turn

    Layer 4: Relationship Rules (evolves with relationship)
        - Intimacy level, permissions, consent protocols
        - Grows over time, never decays (ethical design)

The PersonaEngine assembles these layers into a single system prompt
that is injected at the start of every LLM call.
"""

from __future__ import annotations

from app.persona.base import (
    ConversationalStyle,
    EmotionalDimension,
    EmotionalState,
    IdentityBase,
    PersonaConfig,
    RelationshipRules,
)
from app.persona.engine import PersonaEngine

__all__ = [
    # Data models
    "IdentityBase",
    "ConversationalStyle",
    "RelationshipRules",
    "EmotionalState",
    "EmotionalDimension",
    "PersonaConfig",
    # Engine
    "PersonaEngine",
]
