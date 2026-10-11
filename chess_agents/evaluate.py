"""Grade a finished game with the Stockfish chess engine (optional).

For every move, Stockfish scores the position before and after it. The drop
in score is how much the move cost the player, measured in "centipawns"
(hundredths of a pawn). Averaging that per player gives a simple, objective
measure of how well each agent played.

Stockfish is a separate program. Download it from https://stockfishchess.org
and either put it on your PATH or set STOCKFISH_PATH to the .exe file. Then:

    python -m chess_agents.evaluate games/<id>.pgn
"""

import os
import shutil
import sys

import chess
import chess.engine
import chess.pgn
from dotenv import load_dotenv

ANALYSIS_DEPTH = 12       # how many plies ahead Stockfish searches per position
SCORE_CAP = 1000          # treat anything beyond +/-10 pawns (including mates) as 10 pawns

# A move that loses at least this many centipawns gets the label.
THRESHOLDS = [("blunder", 200), ("mistake", 100), ("inaccuracy", 50)]


def find_stockfish() -> str | None:
    return os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")


def evaluate(pgn_path: str, engine_path: str) -> dict:
    """Return per-player statistics for the game in `pgn_path`."""
    with open(pgn_path, encoding="utf-8") as file:
        game = chess.pgn.read_game(file)
    if game is None:
        raise ValueError(f"No game found in {pgn_path}")

    moves = list(game.mainline_moves())
    board = game.board()

    # Score every position once, always from White's point of view.
    scores = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for move in [None, *moves]:
            if move is not None:
                board.push(move)
            info = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
            score = info["score"].white().score(mate_score=SCORE_CAP)
            scores.append(max(-SCORE_CAP, min(SCORE_CAP, score)))

    stats = {
        color: {"moves": 0, "total_loss": 0, "blunder": 0, "mistake": 0, "inaccuracy": 0}
        for color in ("white", "black")
    }
    for index in range(len(moves)):
        white_moved = index % 2 == 0
        color = "white" if white_moved else "black"
        # How far the score moved against the player who just moved.
        change = scores[index + 1] - scores[index]
        loss = max(0, -change if white_moved else change)

        stats[color]["moves"] += 1
        stats[color]["total_loss"] += loss
        for label, threshold in THRESHOLDS:
            if loss >= threshold:
                stats[color][label] += 1
                break

    for color, player in stats.items():
        player["name"] = game.headers.get(color.capitalize(), color)
        player["average_loss"] = round(player["total_loss"] / player["moves"], 1) if player["moves"] else 0.0
    return stats


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m chess_agents.evaluate <game.pgn>")
    load_dotenv()   # so STOCKFISH_PATH can be set in the .env file
    engine_path = find_stockfish()
    if engine_path is None:
        sys.exit("Stockfish not found. Install it and put it on PATH, or set STOCKFISH_PATH.")

    stats = evaluate(sys.argv[1], engine_path)
    print("Average centipawn loss per move (lower is better):\n")
    for color in ("white", "black"):
        player = stats[color]
        print(f"{color.capitalize()}: {player['name']}")
        print(f"  moves: {player['moves']}   average loss: {player['average_loss']}")
        print(f"  inaccuracies: {player['inaccuracy']}   mistakes: {player['mistake']}   blunders: {player['blunder']}\n")


if __name__ == "__main__":
    main()
