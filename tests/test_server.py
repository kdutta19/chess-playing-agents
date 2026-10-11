"""End-to-end test of the web server, using random movers so no API key is needed."""

import json

import pytest
from fastapi.testclient import TestClient

from chess_agents import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Write saved games to a temporary folder instead of the real games/ folder.
    monkeypatch.setattr(server, "SAVE_DIR", tmp_path)
    monkeypatch.setattr(server, "games", {})
    with TestClient(server.app) as test_client:
        yield test_client


RANDOM_PLAYER = {"model": "random", "persona": "balanced", "effort": "low"}


def read_events(client, game_id):
    """Read the event stream until the server closes it at the end of the game."""
    events = []
    with client.stream("GET", f"/api/games/{game_id}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


def test_page_and_options_load(client):
    assert "Claude chess agents" in client.get("/").text
    options = client.get("/api/options").json()
    assert "claude-opus-5-5" in options["models"]
    assert options["current_game"] is None


def test_a_game_can_be_played_watched_and_resumed(client):
    started = client.post("/api/games", json={
        "white": RANDOM_PLAYER, "black": RANDOM_PLAYER, "max_plies": 2,
    })
    assert started.status_code == 200
    game_id = started.json()["id"]

    events = read_events(client, game_id)
    assert events[0]["type"] == "game_started"
    assert events[-1]["type"] == "game_over"
    assert len([event for event in events if event["type"] == "move_made"]) == 2

    # The board image and the PGN download are available.
    board = client.get(f"/api/games/{game_id}/board.svg?ply=1")
    assert board.headers["content-type"] == "image/svg+xml"
    assert "1/2-1/2" in client.get(f"/api/games/{game_id}/pgn").text

    # The finished game is listed and can be continued as a new game.
    options = client.get("/api/options").json()
    assert options["current_game"] == game_id
    assert options["saved_games"][0]["id"] == game_id

    resumed = client.post("/api/games", json={
        "white": RANDOM_PLAYER, "black": RANDOM_PLAYER, "max_plies": 3, "resume_from": game_id,
    })
    resumed_events = read_events(client, resumed.json()["id"])
    assert len(resumed_events[0]["moves"]) == 2
    assert len([event for event in resumed_events if event["type"] == "move_made"]) == 1


def test_bad_requests_are_rejected(client):
    unknown_model = {"white": {**RANDOM_PLAYER, "model": "gpt-9"}, "black": RANDOM_PLAYER}
    assert client.post("/api/games", json=unknown_model).status_code == 400
    assert client.post("/api/games", json={"resume_from": "../secret"}).status_code == 404
    assert client.get("/api/games/nope/events").status_code == 404
