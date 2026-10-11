"""The tools an agent can call.

Two things live here:

1. TOOL_SCHEMAS: the descriptions sent to Claude so it knows which tools
   exist and what arguments they take.
2. The read-only tool handlers, which turn the game state into text for the
   model.

The one tool that changes the game, make_move, is described here but
executed by the orchestrator, because accepting a move also means checking
whose turn it is and ending the turn.
"""

import json
from dataclasses import dataclass

from .game import ChessGame


@dataclass
class ToolResult:
    """What a tool call sends back to the agent."""

    content: str               # text the model reads
    is_error: bool = False     # True tells the model the call failed
    ends_turn: bool = False    # True once a move has been accepted


# A tool that takes no arguments still needs an (empty) JSON schema.
NO_ARGUMENTS = {"type": "object", "properties": {}}

TOOL_SCHEMAS = [
    {
        "name": "get_board_state",
        "description": (
            "Get the current position: whose turn it is, the FEN string, a "
            "text drawing of the board and the moves played so far. The same "
            "information is given at the start of your turn, so call this "
            "only if you want to look at the board again."
        ),
        "input_schema": NO_ARGUMENTS,
    },
    {
        "name": "get_legal_moves",
        "description": (
            "List every legal move in the current position, in SAN (for "
            "example Nf3) and UCI (for example g1f3). Call this before "
            "moving if you are not certain a move is legal."
        ),
        "input_schema": NO_ARGUMENTS,
    },
    {
        "name": "analyze_position",
        "description": (
            "Get objective facts about the position: the material count for "
            "each side, whether the side to move is in check, pieces that are "
            "attacked and undefended, and the captures and checks available "
            "right now. It reports facts only and does not recommend a move."
        ),
        "input_schema": NO_ARGUMENTS,
    },
    {
        "name": "make_move",
        "description": (
            "Play your move. This ends your turn, so call it exactly once, "
            "after you have decided. If the move is illegal you get an error "
            "and may try again."
        ),
        # strict makes the API guarantee the arguments match this schema,
        # so both fields are always present and are always strings.
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": "The move in SAN (Nf3, exd5, O-O, e8=Q) or UCI (g1f3, e7e8q).",
                },
                "plan": {
                    "type": "string",
                    "description": (
                        "A short note to your future self: your plan and what to "
                        "watch for. You will be shown this note on your next turn, "
                        "and it is the only thing you remember between turns."
                    ),
                },
            },
            "required": ["move", "plan"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Read-only tool handlers
# ---------------------------------------------------------------------------

def get_board_state(game: ChessGame) -> ToolResult:
    text = (
        f"Side to move: {game.turn}\n"
        f"FEN: {game.fen()}\n"
        f"Board (White at the bottom, uppercase = White, lowercase = Black):\n"
        f"{game.ascii_board()}\n"
        f"Moves so far: {game.history_text() or '(none)'}"
    )
    return ToolResult(text)


def get_legal_moves(game: ChessGame) -> ToolResult:
    return ToolResult(json.dumps(game.legal_moves()))


def analyze_position(game: ChessGame) -> ToolResult:
    return ToolResult(json.dumps(game.analysis(), indent=2))


READ_ONLY_TOOLS = {
    "get_board_state": get_board_state,
    "get_legal_moves": get_legal_moves,
    "analyze_position": analyze_position,
}
