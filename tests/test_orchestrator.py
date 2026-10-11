"""Tests for the referee: full games, forfeits, tool gating and the controls."""

import asyncio
import json

import pytest

from chess_agents.agent import AgentError, RandomAgent, Usage
from chess_agents.config import GameLimits
from chess_agents.orchestrator import Orchestrator


class ScriptedAgent:
    """A test player whose turn is a function we write in the test."""

    def __init__(self, color, script):
        self.name, self.color = f"Scripted {color}", color
        self.model, self.persona, self.effort = "random", "test", "-"
        self.usage = Usage()
        self.script = script

    async def take_turn(self, prompt, execute_tool, emit):
        await self.script(prompt, execute_tool)


async def play_first_legal_move(prompt, execute_tool):
    legal = json.loads((await execute_tool("get_legal_moves", {})).content)
    await execute_tool("make_move", {"move": legal[0]["uci"], "plan": "first legal move"})


@pytest.fixture
def limits():
    return GameLimits(random_delay=0)   # no waiting in tests


def make_game(tmp_path, limits, white, black, **options):
    return Orchestrator("test-game", white, black, limits, save_dir=tmp_path, **options)


def events_of(game, event_type):
    return [event for event in game.events if event["type"] == event_type]


# ----- complete games -----------------------------------------------------------

async def test_two_random_players_finish_a_game(tmp_path, limits):
    game = make_game(
        tmp_path, limits,
        RandomAgent("One", "white", "random", limits),
        RandomAgent("Two", "black", "random", limits),
    )
    await game.run()

    assert game.result is not None
    assert game.events[0]["type"] == "game_started"
    assert game.events[-1]["type"] == "game_over"
    assert len(events_of(game, "move_made")) == game.game.ply
    # Both files are written, and the JSON matches the game that was played.
    saved = json.loads((tmp_path / "test-game.json").read_text())
    assert saved["moves"] == game.game.moves_uci()
    assert saved["result"] == game.result
    assert (tmp_path / "test-game.pgn").exists()


async def test_move_limit_ends_the_game_as_a_draw(tmp_path):
    limits = GameLimits(max_plies=4, random_delay=0)
    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", play_first_legal_move),
        ScriptedAgent("black", play_first_legal_move),
    )
    await game.run()

    assert game.game.ply == 4
    assert game.result["result"] == "1/2-1/2"
    assert "move limit" in game.result["reason"]


async def test_notes_are_shown_to_the_player_on_its_next_turn(tmp_path):
    prompts = []

    async def remember_prompt(prompt, execute_tool):
        prompts.append(prompt)
        await play_first_legal_move(prompt, execute_tool)

    limits = GameLimits(max_plies=4, random_delay=0)
    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", remember_prompt),
        ScriptedAgent("black", play_first_legal_move),
    )
    await game.run()

    assert "Your note from your last turn: (none)" in prompts[0]
    assert "Your note from your last turn: first legal move" in prompts[1]


# ----- forfeits and aborts --------------------------------------------------------

async def test_repeated_illegal_moves_forfeit_the_game(tmp_path, limits):
    async def keep_trying_illegal_moves(prompt, execute_tool):
        for _ in range(10):
            await execute_tool("make_move", {"move": "e2e5", "plan": ""})

    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", keep_trying_illegal_moves),
        ScriptedAgent("black", play_first_legal_move),
    )
    await game.run()

    assert game.result["winner"] == "black"
    assert game.result["result"] == "0-1"
    assert "illegal moves" in game.result["reason"]
    assert game.game.ply == 0
    # The first rejections were reported back to the agent as errors.
    assert events_of(game, "tool_result")[0]["is_error"] is True


async def test_an_illegal_move_can_be_corrected(tmp_path):
    async def stumble_then_recover(prompt, execute_tool):
        rejected = await execute_tool("make_move", {"move": "Ke2", "plan": ""})
        assert rejected.is_error and not rejected.ends_turn
        accepted = await execute_tool("make_move", {"move": "e4", "plan": ""})
        assert accepted.ends_turn

    limits = GameLimits(max_plies=1, random_delay=0)
    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", stumble_then_recover),
        ScriptedAgent("black", play_first_legal_move),
    )
    await game.run()

    assert game.game.moves_san() == ["e4"]


async def test_too_many_tool_calls_forfeit_the_game(tmp_path, limits):
    async def look_forever(prompt, execute_tool):
        while True:
            await execute_tool("get_board_state", {})

    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", look_forever),
        ScriptedAgent("black", play_first_legal_move),
    )
    await game.run()

    assert game.result["winner"] == "black"
    assert "tool calls" in game.result["reason"]
    assert len(events_of(game, "tool_call")) == limits.tool_budget


async def test_ending_a_turn_without_moving_forfeits(tmp_path, limits):
    async def do_nothing(prompt, execute_tool):
        pass

    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", do_nothing),
        ScriptedAgent("black", play_first_legal_move),
    )
    await game.run()

    assert game.result["winner"] == "black"
    assert "without moving" in game.result["reason"]


async def test_a_slow_player_forfeits_on_time(tmp_path):
    async def sleep_too_long(prompt, execute_tool):
        await asyncio.sleep(5)

    limits = GameLimits(turn_timeout=0.05, random_delay=0)
    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", sleep_too_long),
        ScriptedAgent("black", play_first_legal_move),
    )
    await game.run()

    assert game.result["winner"] == "black"
    assert "took longer" in game.result["reason"]


async def test_an_api_failure_aborts_without_a_winner(tmp_path, limits):
    async def api_is_down(prompt, execute_tool):
        raise AgentError("could not reach the Claude API")

    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", api_is_down),
        ScriptedAgent("black", play_first_legal_move),
    )
    await game.run()

    assert game.result["winner"] is None
    assert game.result["result"] == "*"
    assert "could not reach" in game.result["reason"]


# ----- tool gating --------------------------------------------------------------------

async def test_a_player_cannot_move_twice_in_one_turn(tmp_path):
    second_attempts = []

    async def move_twice(prompt, execute_tool):
        await execute_tool("make_move", {"move": "e4", "plan": ""})
        second_attempts.append(await execute_tool("make_move", {"move": "d4", "plan": ""}))

    limits = GameLimits(max_plies=1, random_delay=0)
    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", move_twice),
        ScriptedAgent("black", play_first_legal_move),
    )
    await game.run()

    assert second_attempts[0].is_error
    assert game.game.moves_san() == ["e4"]


# ----- resume, pause, step and stop ---------------------------------------------------

async def test_a_saved_game_can_be_resumed(tmp_path):
    limits = GameLimits(max_plies=3, random_delay=0)
    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", play_first_legal_move),
        ScriptedAgent("black", play_first_legal_move),
        moves=["e2e4", "e7e5"], notes={"white": "attack f7"},
    )
    await game.run()

    assert game.events[0]["moves"] == ["e4", "e5"]
    assert game.game.ply == 3                       # one new move on top of the two saved ones
    assert game.game.moves_uci()[:2] == ["e2e4", "e7e5"]


async def test_step_plays_exactly_one_move_while_paused(tmp_path, limits):
    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", play_first_legal_move),
        ScriptedAgent("black", play_first_legal_move),
        start_paused=True,
    )
    game.start()
    await asyncio.sleep(0.05)
    assert game.game.ply == 0          # paused: nothing has been played

    game.step()
    await asyncio.sleep(0.05)
    assert game.game.ply == 1          # one move, then paused again

    game.step()
    await asyncio.sleep(0.05)
    assert game.game.ply == 2

    game.stop()
    await asyncio.sleep(0.05)
    assert game.result["reason"] == "stopped by user"
    assert game.events[-1]["type"] == "game_over"


async def test_stream_replays_past_events_and_ends_with_the_game(tmp_path):
    limits = GameLimits(max_plies=2, random_delay=0)
    game = make_game(
        tmp_path, limits,
        ScriptedAgent("white", play_first_legal_move),
        ScriptedAgent("black", play_first_legal_move),
    )
    game.start()

    # A reader that joins while the game runs still sees every event, in order.
    seen = [event async for event in game.stream()]
    assert [event["seq"] for event in seen] == list(range(len(game.events)))
    assert seen[-1]["type"] == "game_over"

    # A reader that reconnects after the end gets only what it missed.
    late = [event async for event in game.stream(start=len(game.events) - 1)]
    assert [event["type"] for event in late] == ["game_over"]
