"""The web server.

It serves the single web page, starts games, and streams each game's events
to the browser. All the game logic lives in the orchestrator; this file only
translates between HTTP and the orchestrator.

Run it with:

    python -m uvicorn chess_agents.server:app
"""

import datetime
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import storage
from .agent import create_agent
from .config import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    DEFAULT_PERSONA,
    EFFORT_LEVELS,
    MODELS,
    PERSONAS,
    SAVE_DIR,
    GameLimits,
)
from .orchestrator import Orchestrator

# Read ANTHROPIC_API_KEY (and anything else) from a .env file if there is one.
load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Claude chess agents")

# Games started since the server came up, by id. They live in memory; the
# saved files in the games/ folder are what survives a restart.
games: dict[str, Orchestrator] = {}


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class PlayerSettings(BaseModel):
    model: str = DEFAULT_MODEL
    persona: str = DEFAULT_PERSONA
    effort: str = DEFAULT_EFFORT


class NewGame(BaseModel):
    white: PlayerSettings = PlayerSettings()
    black: PlayerSettings = PlayerSettings()
    max_plies: int = Field(default=GameLimits().max_plies, ge=2, le=600)
    start_paused: bool = False
    resume_from: str | None = None   # id of a saved game to continue


def check_player(settings: PlayerSettings) -> None:
    """Reject settings that name a model, persona or effort level we do not have."""
    if settings.model not in MODELS:
        raise HTTPException(400, f"Unknown model: {settings.model}")
    if settings.persona not in PERSONAS:
        raise HTTPException(400, f"Unknown persona: {settings.persona}")
    if settings.effort not in EFFORT_LEVELS:
        raise HTTPException(400, f"Unknown effort level: {settings.effort}")


def get_game(game_id: str) -> Orchestrator:
    if game_id not in games:
        raise HTTPException(404, "No such game")
    return games[game_id]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/options")
async def options():
    """Everything the page needs to build its form, plus the game to show on load."""
    return {
        "models": {model_id: info.label for model_id, info in MODELS.items()},
        "personas": list(PERSONAS),
        "efforts": EFFORT_LEVELS,
        "defaults": {
            "model": DEFAULT_MODEL,
            "persona": DEFAULT_PERSONA,
            "effort": DEFAULT_EFFORT,
            "max_plies": GameLimits().max_plies,
        },
        "saved_games": storage.list_games(SAVE_DIR),
        # The most recently started game, so a page refresh shows it again.
        "current_game": next(reversed(games), None),
    }


@app.post("/api/games")
async def start_game(body: NewGame):
    check_player(body.white)
    check_player(body.black)

    # When resuming, start from the saved moves and the players' saved notes.
    moves, notes = None, None
    if body.resume_from:
        saved = storage.load_game(SAVE_DIR, body.resume_from)
        if saved is None:
            raise HTTPException(404, "No such saved game")
        moves, notes = saved["moves"], saved["notes"]

    limits = GameLimits(max_plies=body.max_plies)
    white = create_agent("Chess Player1", "white", body.white.model, body.white.persona, body.white.effort, limits)
    black = create_agent("Chess Player2", "black", body.black.model, body.black.persona, body.black.effort, limits)

    # A readable, unique id: the time plus a few random characters.
    game_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    game = Orchestrator(
        game_id, white, black, limits, save_dir=SAVE_DIR,
        moves=moves, notes=notes, start_paused=body.start_paused,
    )
    games[game_id] = game
    game.start()
    return {"id": game_id}


@app.get("/api/games/{game_id}/events")
async def events(game_id: str, request: Request):
    """Stream the game's events to the browser as Server-Sent Events (SSE).

    SSE is a long-lived HTTP response in which the server writes one small
    text message per event. The browser's built-in EventSource reads it.
    """
    game = get_game(game_id)

    # If the connection drops, the browser reconnects and tells us the last
    # event it saw, so we carry on from the one after it.
    last_seen = request.headers.get("last-event-id")
    start = int(last_seen) + 1 if last_seen and last_seen.isdigit() else 0

    async def event_stream():
        async for event in game.stream(start):
            yield f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/games/{game_id}/{action}")
async def control(game_id: str, action: str):
    """Pause, resume, step or stop a game."""
    game = get_game(game_id)
    actions = {"pause": game.pause, "resume": game.resume, "step": game.step, "stop": game.stop}
    if action not in actions:
        raise HTTPException(404, f"Unknown action: {action}")
    actions[action]()
    return {"ok": True}


@app.get("/api/games/{game_id}/board.svg")
async def board(game_id: str, ply: int | None = None):
    """The board as an image, after `ply` moves (default: the latest position)."""
    return Response(get_game(game_id).game.svg(ply), media_type="image/svg+xml")


@app.get("/api/games/{game_id}/pgn")
async def pgn(game_id: str):
    """Download the finished game as a PGN file."""
    path = SAVE_DIR / f"{get_game(game_id).game_id}.pgn"
    if not path.exists():
        raise HTTPException(404, "The PGN is written when the game ends")
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )
