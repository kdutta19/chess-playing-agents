"""Saving games to disk and loading them back.

Each game is written as two files in the save folder:

    <id>.json   everything needed to resume the game (moves, notes, settings)
    <id>.pgn    the moves in standard chess format, written when the game ends
"""

import json
from pathlib import Path


def save_game(folder: Path, game_id: str, data: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{game_id}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_pgn(folder: Path, game_id: str, pgn: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{game_id}.pgn").write_text(pgn + "\n", encoding="utf-8")


def load_game(folder: Path, game_id: str) -> dict | None:
    """Load a saved game, or return None if there is no game with that id."""
    # Looking the id up in the folder listing (instead of building a path
    # from it) means an id sent by the browser can never point outside the folder.
    if game_id not in {path.stem for path in folder.glob("*.json")}:
        return None
    return json.loads((folder / f"{game_id}.json").read_text(encoding="utf-8"))


def list_games(folder: Path) -> list[dict]:
    """A short summary of every saved game, newest first."""
    paths = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    summaries = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = data.get("result")
        summaries.append({
            "id": data["id"],
            "plies": len(data["moves"]),
            "white": data["players"]["white"]["model"],
            "black": data["players"]["black"]["model"],
            "status": result["result"] if result else "unfinished",
        })
    return summaries
