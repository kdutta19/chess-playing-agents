"""Tests for the board wrapper. No agents and no network involved."""

import pytest

from chess_agents.game import ChessGame


def test_new_game_starts_with_white_to_move():
    game = ChessGame()
    assert game.turn == "white"
    assert game.ply == 0
    assert len(game.legal_moves()) == 20
    assert not game.is_over()


def test_moves_are_accepted_in_san_and_uci():
    game = ChessGame()
    assert game.make_move("e4") == "e4"        # SAN
    assert game.make_move("e7e5") == "e5"      # UCI
    assert game.make_move("Nf3") == "Nf3"
    assert game.turn == "black"
    assert game.moves_uci() == ["e2e4", "e7e5", "g1f3"]
    assert game.history_text() == "1. e4 e5 2. Nf3"


@pytest.mark.parametrize("bad_move", ["e5", "e2e5", "Ke2", "hello", ""])
def test_illegal_or_unreadable_moves_are_rejected(bad_move):
    game = ChessGame()
    with pytest.raises(ValueError):
        game.make_move(bad_move)
    assert game.ply == 0   # the board is untouched


def test_checkmate_ends_the_game():
    # Fool's mate: the fastest possible checkmate.
    game = ChessGame()
    for move in ["f3", "e5", "g4", "Qh4#"]:
        game.make_move(move)
    assert game.is_over()
    assert game.result() == {"winner": "black", "reason": "checkmate", "result": "0-1"}


def test_threefold_repetition_ends_the_game():
    game = ChessGame()
    for _ in range(2):
        for move in ["Nf3", "Nf6", "Ng1", "Ng8"]:
            game.make_move(move)
    assert game.is_over()
    assert game.result()["winner"] is None
    assert game.result()["reason"] == "threefold repetition"


def test_game_can_be_rebuilt_from_saved_moves():
    game = ChessGame(["e2e4", "e7e5"])
    assert game.ply == 2
    assert game.turn == "white"
    assert game.moves_san() == ["e4", "e5"]


def test_analysis_reports_material_and_hanging_pieces():
    # After 1. e4 d5 the pawns attack each other. Black's d5 pawn is
    # defended by the queen; White's e4 pawn is defended by nothing.
    game = ChessGame(["e2e4", "d7d5"])
    facts = game.analysis()
    assert facts["material"] == {"white": 39, "black": 39}
    assert facts["hanging_pieces"]["white"] == ["pawn on e4"]
    assert facts["hanging_pieces"]["black"] == []
    assert "exd5" in facts["captures_available"]


def test_svg_can_show_an_earlier_position():
    game = ChessGame(["e2e4", "e7e5"])
    assert game.svg().startswith("<svg")
    assert game.svg(ply=0) != game.svg()   # the start position differs from the current one
    assert game.ply == 2                   # drawing an old position does not change the game
