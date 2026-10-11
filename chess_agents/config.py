"""Settings shared by the whole project: models, prices, personas, limits."""

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    """What we need to know about a model to call it and to price it."""

    label: str                  # name shown in the web page
    input_price: float          # US dollars per million input tokens
    output_price: float         # US dollars per million output tokens
    adaptive_thinking: bool     # accepts thinking={"type": "adaptive"}
    supports_effort: bool       # accepts output_config={"effort": ...}


# "random" is not a Claude model. It is a player that picks a random legal
# move, which lets you run and test everything without an API key.
RANDOM_MODEL = "random"

MODELS: dict[str, ModelInfo] = {
    "claude-opus-5-5": ModelInfo("Claude Opus 5.5", 4.00, 20.00, True, True),
    "claude-sonnet-5-5": ModelInfo("Claude Sonnet 5.5", 2.00, 10.00, True, True),
    # Haiku 4.5 rejects both adaptive thinking and the effort setting, so the
    # agent leaves those two parameters out when it plays with this model.
    "claude-haiku-4-5": ModelInfo("Claude Haiku 4.5", 1.00, 5.00, False, False),
    RANDOM_MODEL: ModelInfo("Random mover (no API)", 0.0, 0.0, False, False),
}

DEFAULT_MODEL = "claude-opus-5-5"

# Cached input tokens are billed relative to the normal input price.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

# How hard the model thinks before answering. Lower is faster and cheaper.
EFFORT_LEVELS = ["low", "medium", "high"]
DEFAULT_EFFORT = "low"

# Upper bound on the length of one Claude response (thinking + text + tool call).
MAX_TOKENS = 16000


# ---------------------------------------------------------------------------
# Personas: a playing style added to the agent's system prompt
# ---------------------------------------------------------------------------

PERSONAS: dict[str, str] = {
    "balanced": (
        "Play sound, principled chess: develop your pieces, control the "
        "centre, keep your king safe, and take material when it is free."
    ),
    "aggressive": (
        "You are an attacking player. Prefer open positions, piece activity "
        "and direct threats against the enemy king, and accept some risk to "
        "keep the initiative."
    ),
    "positional": (
        "You are a patient positional player. Prefer solid pawn structure, "
        "good squares for your pieces and small lasting advantages over "
        "speculative attacks."
    ),
}

DEFAULT_PERSONA = "balanced"


# ---------------------------------------------------------------------------
# Game limits enforced by the orchestrator
# ---------------------------------------------------------------------------

@dataclass
class GameLimits:
    max_plies: int = 150          # a ply is one move by one side; 150 = 75 moves each
    tool_budget: int = 8          # tool calls a player may make in one turn
    max_illegal_moves: int = 3    # illegal move attempts in one turn before forfeit
    turn_timeout: float = 180.0   # seconds a player may spend on one turn
    random_delay: float = 0.4     # pause before a random mover plays, so you can watch


# Finished and in-progress games are written here (JSON for resuming, PGN for chess tools).
SAVE_DIR = Path("games")
