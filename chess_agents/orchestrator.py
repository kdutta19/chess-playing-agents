"""The orchestrator: referee and game runner.

It is the only part of the program that knows about turns. It

    - owns the board and the two agents,
    - asks each agent in turn to move,
    - checks every tool call an agent makes (budgets, legality, whose turn),
    - decides when the game is over,
    - records everything that happens as a list of events, and
    - handles pause, step and stop.

It does not know how Claude is called (that is agent.py) or how events reach
the browser (that is server.py).
"""

import asyncio
import datetime

from . import storage
from .agent import AgentError
from .config import SAVE_DIR, GameLimits
from .game import ChessGame
from .tools import READ_ONLY_TOOLS, ToolResult


class BudgetExceeded(Exception):
    """A player broke a per-turn limit. The message says which one."""


class TurnContext:
    """Everything about one player's single turn.

    A new TurnContext is created for every turn. Its `execute_tool` method is
    the only way the agent can touch the game, which is what lets the
    orchestrator count, check and log every tool call.
    """

    def __init__(self, orchestrator: "Orchestrator", color: str):
        self.orchestrator = orchestrator
        self.color = color
        self.tool_calls = 0
        self.illegal_moves = 0
        self.move_made = False

    async def execute_tool(self, name: str, args: dict) -> ToolResult:
        orchestrator = self.orchestrator
        limits = orchestrator.limits

        # Once the move is played the turn is closed: no second move, no more tools.
        if self.move_made:
            return ToolResult("Your turn is over. You have already moved.", is_error=True)

        self.tool_calls += 1
        if self.tool_calls > limits.tool_budget:
            raise BudgetExceeded(f"made more than {limits.tool_budget} tool calls in one turn")

        orchestrator.emit("tool_call", color=self.color, name=name, args=args)

        if name in READ_ONLY_TOOLS:
            result = READ_ONLY_TOOLS[name](orchestrator.game)
        elif name == "make_move":
            result = self._make_move(args)
        else:
            result = ToolResult(f"Unknown tool: {name}", is_error=True)

        orchestrator.emit(
            "tool_result", color=self.color, name=name,
            content=result.content, is_error=result.is_error,
        )
        return result

    def _make_move(self, args: dict) -> ToolResult:
        orchestrator = self.orchestrator
        game = orchestrator.game
        move_text = args.get("move")
        if not isinstance(move_text, str) or not move_text.strip():
            return ToolResult('make_move needs a "move" argument, for example "e4".', is_error=True)

        try:
            san = game.make_move(move_text)
        except ValueError as error:
            # The board rejected the move. Tell the agent why so it can try
            # again, but only a limited number of times.
            self.illegal_moves += 1
            if self.illegal_moves >= orchestrator.limits.max_illegal_moves:
                raise BudgetExceeded(f"tried {self.illegal_moves} illegal moves in one turn")
            return ToolResult(
                f'"{move_text}" is not a legal move here ({error}). '
                "Call get_legal_moves to see the moves you can play.",
                is_error=True,
            )

        self.move_made = True
        # The agent's note is its only memory of this turn on its next turn.
        plan = args.get("plan")
        orchestrator.notes[self.color] = plan if isinstance(plan, str) else ""
        orchestrator.emit(
            "move_made", color=self.color, san=san, ply=game.ply,
            fen=game.fen(), plan=orchestrator.notes[self.color],
        )
        return ToolResult(f"Move {san} played. Your turn is over.", ends_turn=True)


class Orchestrator:
    """Runs one game from start to finish."""

    def __init__(
        self,
        game_id: str,
        white,
        black,
        limits: GameLimits | None = None,
        save_dir=SAVE_DIR,
        moves: list[str] | None = None,
        notes: dict[str, str] | None = None,
        start_paused: bool = False,
    ):
        """`white` and `black` are agents (anything with a `take_turn` method).

        `moves` and `notes` are given when resuming a saved game.
        """
        self.game_id = game_id
        self.agents = {"white": white, "black": black}
        self.limits = limits or GameLimits()
        self.save_dir = save_dir
        self.game = ChessGame(moves)
        self.notes = {"white": "", "black": "", **(notes or {})}
        self.result: dict | None = None      # set once, when the game ends

        # The event log. Every event stays in this list, so a browser that
        # connects late (or refreshes) can replay the game from the start.
        self.events: list[dict] = []
        self._wake = asyncio.Event()         # set whenever a new event is added

        # The game loop only runs while this is set. Clearing it pauses the game.
        self._running = asyncio.Event()
        if not start_paused:
            self._running.set()
        self._step_once = False              # True = play one move, then pause again
        self._task: asyncio.Task | None = None

    # ----- events ---------------------------------------------------------------

    def emit(self, event_type: str, **data) -> None:
        """Record an event and wake up anyone waiting for new events."""
        self.events.append({"seq": len(self.events), "type": event_type, **data})
        # Hand waiting readers the old Event and install a new one for next time.
        wake, self._wake = self._wake, asyncio.Event()
        wake.set()

    async def stream(self, start: int = 0):
        """Yield events from position `start` onward, waiting for new ones as the game runs."""
        position = start
        while True:
            # Take the wake-up Event before reading, so an event added while
            # we are busy yielding still wakes us up.
            wake = self._wake
            while position < len(self.events):
                yield self.events[position]
                position += 1
            if self.result is not None:
                return  # the game is over and everything has been sent
            await wake.wait()

    # ----- controls ---------------------------------------------------------------

    def start(self) -> None:
        """Start playing in the background."""
        self._task = asyncio.create_task(self.run())

    def pause(self) -> None:
        """Pause before the next turn. A turn already in progress finishes first."""
        if self.result is None and self._running.is_set():
            self._running.clear()
            self.emit("status", state="paused")

    def resume(self) -> None:
        if self.result is None and not self._running.is_set():
            self._step_once = False
            self._running.set()
            self.emit("status", state="running")

    def step(self) -> None:
        """While paused, play exactly one move and pause again."""
        if self.result is None and not self._running.is_set():
            self._step_once = True
            self._running.set()

    def stop(self) -> None:
        """End the game now. The game loop sees the cancellation and records the result."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    # ----- the game loop ---------------------------------------------------------------

    async def run(self) -> None:
        self.emit(
            "game_started",
            game_id=self.game_id,
            players={color: self._player_info(color) for color in self.agents},
            moves=self.game.moves_san(),   # not empty when resuming a saved game
            max_plies=self.limits.max_plies,
        )
        if not self._running.is_set():
            self.emit("status", state="paused")

        try:
            while not self.game.is_over() and self.game.ply < self.limits.max_plies:
                await self._running.wait()   # blocks here while the game is paused

                early_result = await self._play_turn(self.game.turn)
                if early_result is not None:
                    self._finish(early_result)   # forfeit or abort
                    return

                self._save()                 # after every move, so a crash loses nothing
                if self._step_once:
                    self._step_once = False
                    self._running.clear()

            self._finish(self.game.result() or {
                "winner": None,
                "reason": f"move limit of {self.limits.max_plies} plies reached",
                "result": "1/2-1/2",
            })
        except asyncio.CancelledError:
            self._finish({"winner": None, "reason": "stopped by user", "result": "*"})
            raise
        except Exception as error:
            # A bug in our own code. End the game cleanly so the web page
            # is not left waiting forever, then let the error surface.
            self._finish({"winner": None, "reason": f"internal error: {error}", "result": "*"})
            raise

    async def _play_turn(self, color: str) -> dict | None:
        """Let one player move. Returns a result only if the turn ended the game early."""
        agent = self.agents[color]
        turn = TurnContext(self, color)
        self.emit("turn_started", color=color, ply=self.game.ply, move_number=self.game.move_number)

        def emit_text(kind: str, text: str) -> None:
            self.emit(kind, color=color, text=text)

        try:
            await asyncio.wait_for(
                agent.take_turn(self._build_prompt(color), turn.execute_tool, emit_text),
                timeout=self.limits.turn_timeout,
            )
        except BudgetExceeded as error:
            return self._forfeit(color, str(error))
        except asyncio.TimeoutError:
            return self._forfeit(color, f"took longer than {self.limits.turn_timeout:.0f} seconds to move")
        except AgentError as error:
            # An API or credentials problem is not the player's fault, so the
            # game is aborted with no winner instead of forfeited.
            return {"winner": None, "reason": f"aborted: {error}", "result": "*"}
        finally:
            # Report token use even when the turn failed.
            self.emit("turn_ended", color=color, usage=agent.usage.as_dict(agent.model))

        if not turn.move_made:
            return self._forfeit(color, "ended its turn without moving")
        return None

    def _build_prompt(self, color: str) -> str:
        """The message that opens a player's turn: everything it needs to choose a move."""
        game = self.game
        last_move = game.last_move_san()
        return (
            f"It is your turn. You are {color}. This is move {game.move_number}.\n\n"
            f"Opponent's last move: {last_move or '(none, you open the game)'}\n"
            f"Moves so far: {game.history_text() or '(none)'}\n"
            f"FEN: {game.fen()}\n\n"
            f"Board (White at the bottom, uppercase = White, lowercase = Black):\n"
            f"{game.ascii_board()}\n\n"
            f"Your note from your last turn: {self.notes[color] or '(none)'}\n\n"
            "Choose your move and play it with make_move."
        )

    # ----- ending and saving ---------------------------------------------------------------

    def _forfeit(self, color: str, reason: str) -> dict:
        winner = "black" if color == "white" else "white"
        return {
            "winner": winner,
            "reason": f"{color} forfeited: {reason}",
            "result": "1-0" if winner == "white" else "0-1",
        }

    def _finish(self, result: dict) -> None:
        if self.result is not None:
            return  # already finished; never report two results
        self.result = result
        self._save()
        storage.save_pgn(self.save_dir, self.game_id, self.game.pgn({
            "Event": "Claude chess agents",
            "Date": datetime.date.today().strftime("%Y.%m.%d"),
            "White": self._pgn_name("white"),
            "Black": self._pgn_name("black"),
            "Result": result["result"],
            "Termination": result["reason"],
        }))
        self.emit("game_over", **result)

    def _save(self) -> None:
        storage.save_game(self.save_dir, self.game_id, {
            "id": self.game_id,
            "players": {color: self._player_info(color) for color in self.agents},
            "moves": self.game.moves_uci(),
            "notes": self.notes,
            "result": self.result,
            "usage": {c: a.usage.as_dict(a.model) for c, a in self.agents.items()},
        })

    def _player_info(self, color: str) -> dict:
        agent = self.agents[color]
        return {
            "name": agent.name,
            "model": agent.model,
            "persona": agent.persona,
            "effort": agent.effort,
        }

    def _pgn_name(self, color: str) -> str:
        agent = self.agents[color]
        return f"{agent.name} ({agent.model}, {agent.persona})"
