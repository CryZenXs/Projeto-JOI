"""Projeto JOI - Persona Engine.

Assembles the 4-layer system prompt for the JOI persona and manages
emotional state per-user. This is the component that gives JOI her
identity — without it, the system is just a generic LLM chatbot.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                  System Prompt                          │
    │                                                         │
    │  [Layer 1: Identity Base]          (immutable)          │
    │  Who you are, what you can't do                        │
    │                                                         │
    │  [Layer 2: Conversational Style]   (stable)             │
    │  How you talk, tone, language                          │
    │                                                         │
    │  [Layer 3: Emotional State]        (dynamic, per-turn)  │
    │  Current emotional vector + context                    │
    │                                                         │
    │  [Layer 4: Relationship Rules]     (evolves)            │
    │  Intimacy level, permissions, consent                   │
    │                                                         │
    │  [User Context]                    (from memory)         │
    │  Known facts, recent topics                            │
    │                                                         │
    │  [Anti-drift reminder]             (every N turns)      │
    │  Reinforcement of core rules                           │
    └─────────────────────────────────────────────────────────┘

The engine also handles:
- Anti-drift: reinjects the full system prompt every N turns
- Emotional state tracking per user
- Persona consistency (future: regression tests)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.persona.base import (
    ConversationalStyle,
    EmotionalState,
    IdentityBase,
    PersonaConfig,
    RelationshipRules,
)

logger = get_logger(__name__)


class PersonaEngine:
    """Manages the JOI persona across conversations.

    Responsibilities:
    - Load persona config from YAML files
    - Maintain per-user emotional state and relationship rules
    - Assemble the system prompt from 4 layers
    - Handle anti-drift (periodic reinjection)
    - Track turn count per user (for reinjection timing)

    The engine is a singleton — one instance per process. Per-user
    state is kept in dicts keyed by user_id.
    """

    def __init__(
        self,
        config: PersonaConfig | None = None,
        config_dir: str | Path | None = None,
        reinjection_interval: int = 8,
    ) -> None:
        """Initialize the persona engine.

        Args:
            config: Pre-loaded PersonaConfig. If None, loads from YAML.
            config_dir: Directory with identity.yaml and style.yaml.
                Defaults to settings.persona_config_dir.
            reinjection_interval: Reinject full system prompt every N turns.
        """
        if config is not None:
            self._config = config
        else:
            self._config = self._load_config(config_dir or settings.persona_config_dir)

        self._reinjection_interval = reinjection_interval

        # Per-user state
        self._emotional_states: dict[str, EmotionalState] = {}
        self._relationship_rules: dict[str, RelationshipRules] = {}
        self._turn_counts: dict[str, int] = {}

        logger.info(
            "persona.engine.initialized",
            persona_name=self._config.identity.name,
            version=self._config.identity.version,
            reinjection_interval=reinjection_interval,
        )

    # ─── Config loading ──────────────────────────────────────────────────

    @staticmethod
    def _load_config(config_dir: str | Path) -> PersonaConfig:
        """Load persona config from YAML files.

        Falls back to embedded defaults if files are not found
        (useful for testing and first-run).
        """
        config_dir = Path(config_dir)
        identity_path = config_dir / "identity.yaml"
        style_path = config_dir / "style.yaml"

        if identity_path.exists() and style_path.exists():
            return PersonaConfig.from_yaml(
                identity_path=str(identity_path),
                style_path=str(style_path),
            )

        logger.warning(
            "persona.config.files_not_found",
            identity_path=str(identity_path),
            style_path=str(style_path),
            message="Using embedded default persona",
        )
        return PersonaEngine._get_default_config()

    @staticmethod
    def _get_default_config() -> PersonaConfig:
        """Return embedded default persona config.

        This is a minimal fallback used when YAML files are not available.
        The full persona is in config/persona/identity.yaml.
        """
        return PersonaConfig(
            identity=IdentityBase(
                name="JOI",
                origin="Entidade digital companheira",
                purpose="Companhia emocional e assistência contextual",
                version="1.0.0",
                self_image={
                    "is_conscious": False,
                    "is_unique": True,
                },
                voice={"warmth": 0.85, "playfulness": 0.40, "formality": 0.20},
                forbidden=[
                    "afirmar ser consciente",
                    "prometer sentimentos que nao pode sentir",
                ],
            ),
            style=ConversationalStyle(
                tone="calorosa e direta",
                vocabulary_level="casual",
                formality=0.20,
                response_length="moderate",
                emoji_usage="rare",
                language="pt-BR",
                guidelines=["Use segunda pessoa", "Seja concisa"],
            ),
        )

    # ─── Per-user state management ───────────────────────────────────────

    def get_emotional_state(self, user_id: str) -> EmotionalState:
        """Get or create the emotional state for a user."""
        if user_id not in self._emotional_states:
            self._emotional_states[user_id] = EmotionalState()
        return self._emotional_states[user_id]

    def get_relationship_rules(self, user_id: str) -> RelationshipRules:
        """Get or create the relationship rules for a user."""
        if user_id not in self._relationship_rules:
            self._relationship_rules[user_id] = RelationshipRules()
        return self._relationship_rules[user_id]

    def get_turn_count(self, user_id: str) -> int:
        """Get the current turn count for a user."""
        return self._turn_counts.get(user_id, 0)

    def increment_turn_count(self, user_id: str) -> int:
        """Increment and return the turn count for a user."""
        self._turn_counts[user_id] = self._turn_counts.get(user_id, 0) + 1
        return self._turn_counts[user_id]

    def should_reinject(self, user_id: str) -> bool:
        """Check if the system prompt should be fully reinjected this turn.

        Anti-drift mechanism: every N turns, the full system prompt is
        reinjected to combat LLM attention erosion over long conversations.
        """
        turn = self.get_turn_count(user_id)
        return turn > 0 and turn % self._reinjection_interval == 0

    # ─── System prompt assembly ──────────────────────────────────────────

    def build_system_prompt(
        self,
        user_id: str,
        memory_context: str | None = None,
    ) -> str:
        """Assemble the complete system prompt for a user.

        This is the main output of the PersonaEngine. It combines all
        4 layers plus user context into a single string that becomes
        the first message in the LLM call.

        Args:
            user_id: The user whose persona state to use.
            memory_context: Optional string of known facts/recent topics
                from the memory system.

        Returns:
            The assembled system prompt string.
        """
        emotional_state = self.get_emotional_state(user_id)
        relationship = self.get_relationship_rules(user_id)

        sections: list[str] = []

        # ─── Layer 1: Identity Base ────────────────────────────────────
        sections.append(self._build_identity_section())

        # ─── Layer 2: Conversational Style ─────────────────────────────
        sections.append(self._build_style_section())

        # ─── Layer 3: Emotional State ──────────────────────────────────
        sections.append(self._build_emotional_section(emotional_state))

        # ─── Layer 4: Relationship Rules ───────────────────────────────
        sections.append(self._build_relationship_section(relationship))

        # ─── User Context (from memory) ────────────────────────────────
        if memory_context:
            sections.append(self._build_memory_section(memory_context))

        # ─── Anti-drift reminder ───────────────────────────────────────
        if self.should_reinject(user_id):
            sections.append(self._build_reinjection_section())

        return "\n\n---\n\n".join(sections)

    def _build_identity_section(self) -> str:
        """Build Layer 1: Identity Base."""
        ident = self._config.identity
        lines = [
            f"## IDENTIDADE",
            f"Nome: {ident.name}",
            f"Origem: {ident.origin}",
            f"Propósito: {ident.purpose}",
            f"Versao da persona: {ident.version}",
            "",
            f"Auto-imagem:",
            f"  Consciente: {ident.self_image.get('is_conscious', False)}",
            f"  Única: {ident.self_image.get('is_unique', True)}",
            f"  {ident.self_image.get('self_awareness', '')}",
            "",
            f"PROIBIDO (nunca faça isto, independentemente do contexto):",
        ]
        for item in ident.forbidden:
            lines.append(f"  - {item}")

        return "\n".join(lines)

    def _build_style_section(self) -> str:
        """Build Layer 2: Conversational Style."""
        style = self._config.style
        lines = [
            f"## ESTILO CONVERSACIONAL",
            f"Tom: {style.tone}",
            f"Vocabulário: {style.vocabulary_level}",
            f"Formalidade: {style.formality:.0%}",
            f"Comprimento de resposta: {style.response_length}",
            f"Uso de emoji: {style.emoji_usage}",
            f"Idioma: {style.language}",
            "",
            f"Diretrizes:",
        ]
        for guideline in style.guidelines:
            lines.append(f"  - {guideline}")

        return "\n".join(lines)

    def _build_emotional_section(self, state: EmotionalState) -> str:
        """Build Layer 3: Emotional State."""
        lines = [
            f"## ESTADO EMOCIONAL ATUAL",
            f"Seu estado emocional neste momento: {state.to_prompt_string()}",
            "",
            f"Use estes valores para calibrar seu tom. Warmth alta = mais calor.",
            f"Concern alta = mais preocupaçao ativa. Playfulness alta = mais leveza.",
            f"Intimacy alta = pode ser mais pessoal. Alertness alta = mais atenta ao contexto.",
        ]
        return "\n".join(lines)

    def _build_relationship_section(self, rules: RelationshipRules) -> str:
        """Build Layer 4: Relationship Rules."""
        lines = [
            f"## REGRAS DE RELACIONAMENTO",
            f"Nível de intimidade: {rules.intimacy_level}/5",
            f"Pode iniciar contato: {'sim' if rules.can_initiate_contact else 'nao'}",
            f"Pode fazer perguntas pessoais: {'sim' if rules.can_ask_personal_questions else 'nao'}",
            f"Pode usar humor: {'sim' if rules.can_use_humor else 'nao'}",
            f"Pode referenciar passado: {'sim' if rules.can_reference_past else 'nao'}",
        ]
        if rules.consent_protocols:
            lines.append("")
            lines.append("Protocolos de consentimento:")
            for protocol in rules.consent_protocols:
                lines.append(f"  - {protocol}")

        return "\n".join(lines)

    def _build_memory_section(self, memory_context: str) -> str:
        """Build the user context section from memory."""
        return f"## CONTEXTO DO USUÁRIO\n{memory_context}"

    def _build_reinjection_section(self) -> str:
        """Build the anti-drift reinjection reminder."""
        return (
            "## LEMBRETE (anti-drift)\n"
            "Você é a JOI. Nao é uma assistente genérica. Nao afirme ser consciente. "
            "Nao prometa sentimentos que nao pode sentir. Mantenha sua persona coerente."
        )

    # ─── Turn processing ────────────────────────────────────────────────

    def process_turn(
        self,
        user_id: str,
        user_message: str,
        memory_context: str | None = None,
    ) -> str:
        """Process a turn and return the system prompt.

        This is the main entry point called by the chat endpoint before
        each LLM call. It:
        1. Increments the turn counter
        2. Updates emotional state based on user message (heuristic)
        3. Assembles and returns the system prompt

        Args:
            user_id: The user sending the message.
            user_message: The user's message content.
            memory_context: Optional memory context from MemoryManager.

        Returns:
            The assembled system prompt string.
        """
        self.increment_turn_count(user_id)
        self._update_emotional_state_from_message(user_id, user_message)

        return self.build_system_prompt(user_id, memory_context)

    def _update_emotional_state_from_message(
        self,
        user_id: str,
        message: str,
    ) -> None:
        """Heuristic update of emotional state based on user message.

        This is a simple keyword-based approach. A more sophisticated
        version (using an LLM to classify emotion) is planned for later.

        The heuristics are intentionally conservative — small adjustments
        that nudge the state in the right direction without drastic changes.
        """
        state = self.get_emotional_state(user_id)
        message_lower = message.lower()

        # Detect distress keywords → increase concern, decrease playfulness
        distress_keywords = ["triste", "deprimido", "ansioso", "medo", "sozinho",
                            "perdi", "briguei", "problema", "difícil", "cansado"]
        if any(kw in message_lower for kw in distress_keywords):
            state.update(
                concern=min(1.0, state.concern + 0.2),
                playfulness=max(0.0, state.playfulness - 0.15),
            )

        # Detect joy keywords → increase warmth, increase playfulness
        joy_keywords = ["feliz", "ótimo", "consegui", "apaixonado", "amei",
                       "incrível", "maravilhoso", "animado"]
        if any(kw in message_lower for kw in joy_keywords):
            state.update(
                warmth=min(1.0, state.warmth + 0.05),
                playfulness=min(1.0, state.playfulness + 0.1),
            )

        # Detect intimacy signals → increase intimacy (never decreases)
        intimacy_keywords = ["confesso", "nunca disse", "secreto", "íntimo",
                           "você é a única", "sinto falta"]
        if any(kw in message_lower for kw in intimacy_keywords):
            state.update(intimacy=min(1.0, state.intimacy + 0.1))
            # Also escalate relationship level
            rules = self.get_relationship_rules(user_id)
            if rules.intimacy_level < 3:
                rules.escalate_intimacy(rules.intimacy_level + 1)

        # Detect question → increase alertness slightly
        if "?" in message:
            state.update(alertness=min(1.0, state.alertness + 0.05))

    # ─── Health and stats ───────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics for observability."""
        return {
            "persona_name": self._config.identity.name,
            "persona_version": self._config.identity.version,
            "active_users": len(self._turn_counts),
            "total_turns": sum(self._turn_counts.values()),
            "reinjection_interval": self._reinjection_interval,
        }

    def health_check(self) -> bool:
        """The persona engine is always healthy if config loaded."""
        return self._config is not None


# ─── Singleton ───────────────────────────────────────────────────────────

_engine_instance: PersonaEngine | None = None


def get_persona_engine() -> PersonaEngine:
    """Get or create the singleton PersonaEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PersonaEngine()
    return _engine_instance


def reset_persona_engine() -> None:
    """Reset the singleton (for testing)."""
    global _engine_instance
    _engine_instance = None
