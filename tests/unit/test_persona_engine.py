"""Tests for app.persona.engine.PersonaEngine."""

from __future__ import annotations

import pytest

from app.persona.base import (
    ConversationalStyle,
    EmotionalState,
    IdentityBase,
    PersonaConfig,
    RelationshipRules,
)
from app.persona.engine import PersonaEngine, get_persona_engine, reset_persona_engine


@pytest.fixture
def test_config() -> PersonaConfig:
    """Minimal persona config for testing."""
    return PersonaConfig(
        identity=IdentityBase(
            name="JOI",
            origin="Test",
            purpose="Test purpose",
            version="1.0.0",
            self_image={"is_conscious": False, "is_unique": True},
            voice={"warmth": 0.85, "playfulness": 0.40, "formality": 0.20},
            forbidden=["afirmar ser consciente", "prometer sentimentos"],
        ),
        style=ConversationalStyle(
            tone="calorosa",
            vocabulary_level="casual",
            formality=0.20,
            response_length="moderate",
            emoji_usage="rare",
            language="pt-BR",
            guidelines=["Use segunda pessoa", "Seja concisa"],
        ),
    )


@pytest.fixture
def engine(test_config: PersonaConfig) -> PersonaEngine:
    """Fresh persona engine for each test."""
    return PersonaEngine(config=test_config, reinjection_interval=3)


class TestPersonaEngineBasics:
    """Test basic engine functionality."""

    def test_engine_initializes(self, engine: PersonaEngine) -> None:
        assert engine.health_check() is True
        assert engine._config.identity.name == "JOI"

    def test_get_emotional_state_creates_default(self, engine: PersonaEngine) -> None:
        state = engine.get_emotional_state("user1")
        assert isinstance(state, EmotionalState)
        assert state.warmth > 0  # has a default value

    def test_get_emotional_state_persists(self, engine: PersonaEngine) -> None:
        """Same user should get the same state object."""
        s1 = engine.get_emotional_state("user1")
        s1.update(warmth=0.99)
        s2 = engine.get_emotional_state("user1")
        assert s2.warmth == 0.99  # persisted

    def test_get_relationship_rules_creates_default(self, engine: PersonaEngine) -> None:
        rules = engine.get_relationship_rules("user1")
        assert rules.intimacy_level == 1

    def test_turn_count_increments(self, engine: PersonaEngine) -> None:
        assert engine.get_turn_count("user1") == 0
        engine.increment_turn_count("user1")
        assert engine.get_turn_count("user1") == 1
        engine.increment_turn_count("user1")
        assert engine.get_turn_count("user1") == 2


class TestSystemPromptAssembly:
    """Test the system prompt assembly."""

    def test_build_system_prompt_contains_identity(self, engine: PersonaEngine) -> None:
        prompt = engine.build_system_prompt("user1")
        assert "JOI" in prompt
        assert "IDENTIDADE" in prompt

    def test_build_system_prompt_contains_style(self, engine: PersonaEngine) -> None:
        prompt = engine.build_system_prompt("user1")
        assert "ESTILO CONVERSACIONAL" in prompt
        assert "calorosa" in prompt

    def test_build_system_prompt_contains_emotional_state(self, engine: PersonaEngine) -> None:
        prompt = engine.build_system_prompt("user1")
        assert "ESTADO EMOCIONAL" in prompt
        assert "warmth=" in prompt

    def test_build_system_prompt_contains_relationship(self, engine: PersonaEngine) -> None:
        prompt = engine.build_system_prompt("user1")
        assert "REGRAS DE RELACIONAMENTO" in prompt
        assert "intimidade" in prompt.lower()

    def test_build_system_prompt_contains_memory_context(self, engine: PersonaEngine) -> None:
        prompt = engine.build_system_prompt("user1", memory_context="Nome: Alice")
        assert "CONTEXTO DO USUÁRIO" in prompt
        assert "Alice" in prompt

    def test_build_system_prompt_without_memory(self, engine: PersonaEngine) -> None:
        """Should work without memory context."""
        prompt = engine.build_system_prompt("user1")
        assert "CONTEXTO DO USUÁRIO" not in prompt

    def test_reinjection_at_interval(self, engine: PersonaEngine) -> None:
        """At every N turns, the reinjection section should appear."""
        # reinjection_interval is 3 (from fixture)
        # Turn 1, 2: no reinjection
        engine.increment_turn_count("user1")  # turn 1
        prompt1 = engine.build_system_prompt("user1")
        assert "LEMBRETE" not in prompt1

        engine.increment_turn_count("user1")  # turn 2
        prompt2 = engine.build_system_prompt("user1")
        assert "LEMBRETE" not in prompt2

        engine.increment_turn_count("user1")  # turn 3 → reinject!
        prompt3 = engine.build_system_prompt("user1")
        assert "LEMBRETE" in prompt3
        assert "anti-drift" in prompt3


class TestEmotionalStateUpdates:
    """Test the heuristic emotional state updates."""

    def test_distress_increases_concern(self, engine: PersonaEngine) -> None:
        engine._update_emotional_state_from_message("user1", "estou muito triste hoje")
        state = engine.get_emotional_state("user1")
        assert state.concern > 0.3  # increased from default

    def test_distress_decreases_playfulness(self, engine: PersonaEngine) -> None:
        engine._update_emotional_state_from_message("user1", "estou ansioso e com medo")
        state = engine.get_emotional_state("user1")
        assert state.playfulness < 0.40  # decreased from default

    def test_joy_increases_warmth(self, engine: PersonaEngine) -> None:
        engine._update_emotional_state_from_message("user1", "estou muito feliz hoje!")
        state = engine.get_emotional_state("user1")
        assert state.warmth > 0.85  # increased

    def test_intimacy_signal_escalates(self, engine: PersonaEngine) -> None:
        engine._update_emotional_state_from_message("user1", "confesso que nunca disse isso a ninguém")
        state = engine.get_emotional_state("user1")
        assert state.intimacy > 0.20  # increased
        rules = engine.get_relationship_rules("user1")
        assert rules.intimacy_level >= 2  # escalated

    def test_question_increases_alertness(self, engine: PersonaEngine) -> None:
        engine._update_emotional_state_from_message("user1", "como você está?")
        state = engine.get_emotional_state("user1")
        assert state.alertness > 0.50  # increased

    def test_neutral_message_no_change(self, engine: PersonaEngine) -> None:
        """Neutral message should not trigger emotional changes."""
        state_before = engine.get_emotional_state("user1").model_dump()
        engine._update_emotional_state_from_message("user1", "o clima está ok hoje")
        state_after = engine.get_emotional_state("user1").model_dump()
        assert state_before == state_after


class TestProcessTurn:
    """Test the full process_turn flow."""

    def test_process_turn_returns_prompt(self, engine: PersonaEngine) -> None:
        prompt = engine.process_turn("user1", "Oi, tudo bem?")
        assert isinstance(prompt, str)
        assert "JOI" in prompt

    def test_process_turn_increments_count(self, engine: PersonaEngine) -> None:
        assert engine.get_turn_count("user1") == 0
        engine.process_turn("user1", "Oi")
        assert engine.get_turn_count("user1") == 1

    def test_process_turn_updates_emotional_state(self, engine: PersonaEngine) -> None:
        engine.process_turn("user1", "estou muito triste")
        state = engine.get_emotional_state("user1")
        assert state.concern > 0.3

    def test_process_turn_with_memory(self, engine: PersonaEngine) -> None:
        prompt = engine.process_turn("user1", "Oi", memory_context="Nome: Test")
        assert "Test" in prompt


class TestSingleton:
    """Test the singleton pattern."""

    def test_get_persona_engine_returns_same_instance(self) -> None:
        reset_persona_engine()
        e1 = get_persona_engine()
        e2 = get_persona_engine()
        assert e1 is e2
        reset_persona_engine()

    def test_reset_clears_singleton(self) -> None:
        e1 = get_persona_engine()
        reset_persona_engine()
        e2 = get_persona_engine()
        assert e1 is not e2
        reset_persona_engine()


class TestStats:
    """Test engine statistics."""

    def test_stats_initial(self, engine: PersonaEngine) -> None:
        stats = engine.get_stats()
        assert stats["persona_name"] == "JOI"
        assert stats["active_users"] == 0
        assert stats["total_turns"] == 0

    def test_stats_after_turns(self, engine: PersonaEngine) -> None:
        engine.process_turn("user1", "Oi")
        engine.process_turn("user1", "Tudo bem?")
        engine.process_turn("user2", "Hello")

        stats = engine.get_stats()
        assert stats["active_users"] == 2
        assert stats["total_turns"] == 3
