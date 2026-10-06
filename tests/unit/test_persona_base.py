"""Tests for app.persona.base (data models)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.persona.base import (
    ConversationalStyle,
    EmotionalDimension,
    EmotionalState,
    IdentityBase,
    PersonaConfig,
    RelationshipRules,
)


class TestEmotionalState:
    """Test the emotional state vector."""

    def test_default_values(self) -> None:
        state = EmotionalState()
        assert 0.0 <= state.warmth <= 1.0
        assert 0.0 <= state.concern <= 1.0
        assert 0.0 <= state.playfulness <= 1.0
        assert 0.0 <= state.intimacy <= 1.0
        assert 0.0 <= state.alertness <= 1.0

    def test_update_changes_values(self) -> None:
        state = EmotionalState()
        state.update(warmth=0.9, concern=0.7)
        assert state.warmth == 0.9
        assert state.concern == 0.7

    def test_update_clamps_to_range(self) -> None:
        state = EmotionalState()
        state.update(warmth=1.5)
        assert state.warmth == 1.0
        state.update(warmth=-0.5)
        assert state.warmth == 0.0

    def test_intimacy_never_decreases(self) -> None:
        """Critical: intimacy can only increase, never decrease."""
        state = EmotionalState(intimacy=0.5)
        state.update(intimacy=0.3)  # try to decrease
        assert state.intimacy == 0.5  # unchanged

    def test_intimacy_can_increase(self) -> None:
        state = EmotionalState(intimacy=0.3)
        state.update(intimacy=0.7)
        assert state.intimacy == 0.7

    def test_decay_reduces_non_intimacy(self) -> None:
        state = EmotionalState(warmth=1.0, concern=1.0, playfulness=1.0,
                               intimacy=0.8, alertness=1.0)
        state.decay(0.5)
        assert state.warmth == 0.5
        assert state.concern == 0.5
        assert state.playfulness == 0.5
        assert state.alertness == 0.5

    def test_decay_does_not_affect_intimacy(self) -> None:
        """Critical: decay must NOT reduce intimacy."""
        state = EmotionalState(intimacy=0.8)
        state.decay(0.1)  # aggressive decay
        assert state.intimacy == 0.8  # unchanged

    def test_to_prompt_string(self) -> None:
        state = EmotionalState(warmth=0.85, concern=0.30, playfulness=0.40,
                               intimacy=0.20, alertness=0.50)
        s = state.to_prompt_string()
        assert "warmth=0.85" in s
        assert "concern=0.30" in s
        assert "intimacy=0.20" in s

    def test_to_dict(self) -> None:
        state = EmotionalState()
        d = state.to_dict()
        assert EmotionalDimension.WARMTH.value in d
        assert EmotionalDimension.INTIMACY.value in d
        assert len(d) == 5


class TestRelationshipRules:
    """Test relationship rules and escalation."""

    def test_default_level_1(self) -> None:
        rules = RelationshipRules()
        assert rules.intimacy_level == 1
        assert rules.can_initiate_contact is False
        assert rules.can_ask_personal_questions is False

    def test_escalate_increases_level(self) -> None:
        rules = RelationshipRules()
        assert rules.escalate_intimacy(2) is True
        assert rules.intimacy_level == 2

    def test_escalate_unlocks_permissions(self) -> None:
        rules = RelationshipRules()
        rules.escalate_intimacy(3)
        assert rules.can_ask_personal_questions is True
        assert rules.can_reference_past is True

    def test_escalate_level_4_unlocks_initiation(self) -> None:
        rules = RelationshipRules()
        rules.escalate_intimacy(4)
        assert rules.can_initiate_contact is True

    def test_escalate_cannot_decrease(self) -> None:
        rules = RelationshipRules(intimacy_level=3)
        assert rules.escalate_intimacy(2) is False
        assert rules.intimacy_level == 3  # unchanged

    def test_escalate_max_level_5(self) -> None:
        rules = RelationshipRules()
        assert rules.escalate_intimacy(6) is False  # above max


class TestIdentityBase:
    """Test identity base model."""

    def test_valid_identity(self) -> None:
        ident = IdentityBase(
            name="JOI",
            origin="Test origin",
            purpose="Test purpose",
            version="1.0.0",
            self_image={"is_conscious": False, "is_unique": True},
            voice={"warmth": 0.85, "playfulness": 0.40, "formality": 0.20},
            forbidden=["afirmar ser consciente"],
        )
        assert ident.name == "JOI"
        assert ident.version == "1.0.0"

    def test_voice_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IdentityBase(
                name="JOI",
                origin="Test",
                purpose="Test",
                version="1.0.0",
                self_image={},
                voice={"warmth": 1.5},  # out of range
                forbidden=[],
            )


class TestPersonaConfig:
    """Test complete persona config."""

    def test_from_yaml(self, tmp_path) -> None:
        import yaml

        identity_data = {
            "name": "JOI",
            "origin": "Test",
            "purpose": "Test purpose",
            "version": "1.0.0",
            "self_image": {"is_conscious": False, "is_unique": True},
            "voice": {"warmth": 0.8, "playfulness": 0.4, "formality": 0.2},
            "forbidden": ["regra1"],
        }
        style_data = {
            "tone": "calorosa",
            "vocabulary_level": "casual",
            "formality": 0.2,
            "response_length": "moderate",
            "emoji_usage": "rare",
            "language": "pt-BR",
            "guidelines": ["regra1"],
        }

        identity_path = tmp_path / "identity.yaml"
        style_path = tmp_path / "style.yaml"
        identity_path.write_text(yaml.dump(identity_data), encoding="utf-8")
        style_path.write_text(yaml.dump(style_data), encoding="utf-8")

        config = PersonaConfig.from_yaml(
            identity_path=str(identity_path),
            style_path=str(style_path),
        )

        assert config.identity.name == "JOI"
        assert config.style.tone == "calorosa"
