"""The chess board and the rules.

This is the single source of truth for the game. The agents never hold a
board of their own: they can only read this one and submit moves to it
through tools, so a model that imagines an illegal move cannot corrupt the
game. All chess knowledge comes from the python-chess library.
"""

import re

import chess
import chess.pgn
import chess.svg

# Standard piece values, used for the material count in analysis().
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# A move in UCI notation: from-square, to-square, optional promotion piece (e2e4, e7e8q).
UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


class ChessGame:
    """A game in progress."""

    def __init__(self, moves: list[str] | None = None):
        """Start a new game, or rebuild one from a list of UCI moves (used when resuming)."""
        self.board = chess.Board()
        for uci in moves or []:
            self.board.push_uci(uci)

    # ----- reading the position ------------------------------------------

    @property
    def turn(self) -> str:
        """The side to move: "white" or "black"."""
        return color_name(self.board.turn)

    @property
    def ply(self) -> int:
        """How many single moves have been played so far."""
        return len(self.board.move_stack)

    @property
    def move_number(self) -> int:
        """The move number as chess players count it (both sides share a number)."""
        return self.board.fullmove_number

    def fen(self) -> str:
        """The position as a FEN string, the standard one-line board notation."""
        return self.board.fen()

    def ascii_board(self) -> str:
        """A text drawing of the board with rank and file labels, White at the bottom."""
        rows = str(self.board).splitlines()
        lines = [f"{8 - i} {row}" for i, row in enumerate(rows)]
        lines.append("  a b c d e f g h")
        return "\n".join(lines)

    def moves_uci(self) -> list[str]:
        return [move.uci() for move in self.board.move_stack]

    def moves_san(self) -> list[str]:
        """Every move played so far in SAN, the notation humans read (Nf3, exd5, O-O)."""
        replay = chess.Board()
        san_moves = []
        for move in self.board.move_stack:
            san_moves.append(replay.san(move))
            replay.push(move)
        return san_moves

    def history_text(self) -> str:
        """The game so far as numbered moves, e.g. "1. e4 e5 2. Nf3"."""
        return chess.Board().variation_san(self.board.move_stack)

    def last_move_san(self) -> str | None:
        san_moves = self.moves_san()
        return san_moves[-1] if san_moves else None

    def legal_moves(self) -> list[dict]:
        """Every legal move for the side to move, in both notations."""
        return [
            {"uci": move.uci(), "san": self.board.san(move)}
            for move in self.board.legal_moves
        ]

    # ----- making a move ---------------------------------------------------

    def parse_move(self, text: str) -> chess.Move:
        """Turn a move written in UCI or SAN into a legal move.

        Raises ValueError (with a readable message) if the text is not a
        move, or is a move that is not legal in the current position.
        """
        text = text.strip()
        if UCI_PATTERN.match(text.lower()):
            return self.board.parse_uci(text.lower())
        return self.board.parse_san(text)

    def make_move(self, text: str) -> str:
        """Play a move and return it in SAN. Raises ValueError if it is illegal."""
        move = self.parse_move(text)
        san = self.board.san(move)
        self.board.push(move)
        return san

    # ----- how the game ends -------------------------------------------------

    def is_over(self) -> bool:
        # claim_draw=True also ends the game on threefold repetition and the
        # fifty-move rule, which otherwise need a player to claim them.
        return self.board.is_game_over(claim_draw=True)

    def result(self) -> dict | None:
        """The winner and the reason once the game is over, otherwise None."""
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return None
        winner = None if outcome.winner is None else color_name(outcome.winner)
        return {
            "winner": winner,
            "reason": outcome.termination.name.replace("_", " ").lower(),
            "result": outcome.result(),  # "1-0", "0-1" or "1/2-1/2"
        }

    # ----- facts an agent can reason over -----------------------------------------

    def analysis(self) -> dict:
        """Simple, objective facts about the position.

        This is not a chess engine and gives no advice. It only saves the
        model from counting pieces and attackers by hand, which language
        models do unreliably.
        """
        board = self.board
        material = {"white": 0, "black": 0}
        hanging = {"white": [], "black": []}

        for square, piece in board.piece_map().items():
            side = color_name(piece.color)
            material[side] += PIECE_VALUES.get(piece.piece_type, 0)

            # A piece is "hanging" if the opponent attacks it and nothing defends it.
            attacked = board.is_attacked_by(not piece.color, square)
            defended = board.is_attacked_by(piece.color, square)
            if attacked and not defended and piece.piece_type != chess.KING:
                name = chess.piece_name(piece.piece_type)
                hanging[side].append(f"{name} on {chess.square_name(square)}")

        legal = list(board.legal_moves)
        return {
            "side_to_move": self.turn,
            "in_check": board.is_check(),
            "material": material,
            "hanging_pieces": hanging,
            "captures_available": [board.san(m) for m in legal if board.is_capture(m)],
            "checks_available": [board.san(m) for m in legal if board.gives_check(m)],
        }

    # ----- output for people -----------------------------------------------------

    def svg(self, ply: int | None = None) -> str:
        """Draw the board as an SVG image, after `ply` moves (default: the current position)."""
        board = self.board
        if ply is not None and ply < self.ply:
            # Replay only the first `ply` moves to show an earlier position.
            board = chess.Board()
            for move in self.board.move_stack[:ply]:
                board.push(move)

        last_move = board.peek() if board.move_stack else None
        king_in_check = board.king(board.turn) if board.is_check() else None
        return chess.svg.board(board, lastmove=last_move, check=king_in_check)

    def pgn(self, headers: dict[str, str]) -> str:
        """The game in PGN, the standard file format other chess programs can open."""
        game = chess.pgn.Game.from_board(self.board)
        for key, value in headers.items():
            game.headers[key] = value
        return str(game)
