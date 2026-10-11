# Two chess players (Claude agents)

Two AI agents, **Chess Player1** (White) and **Chess Player2** (Black), play a
full game of chess against each other while you watch in a web page. Each agent
is a Claude model that looks at the board, calls tools, explains its thinking
and plays a move.

The project is a small, readable demonstration of how agentic AI systems are
built: the agent loop, tool use, guardrails, memory, orchestration and
observability are all written by hand, with no agent framework.

## How it works

```
                 ┌──────────────────────────────┐
   browser  ◄────┤  server.py  (FastAPI + SSE)  │
                 └──────────────┬───────────────┘
                                │ events
                 ┌──────────────▼───────────────┐
                 │  orchestrator.py  (referee)  │
                 │  turn order · budgets · log  │
                 └───────┬──────────────┬───────┘
                 your turn              your turn
                 ┌───────▼─────┐  ┌─────▼───────┐
                 │  agent.py   │  │  agent.py   │
                 │  Player1    │  │  Player2    │
                 └───────┬─────┘  └─────┬───────┘
                         └──────┬───────┘
                         tool calls (checked by the referee)
                 ┌──────────────▼───────────────┐
                 │  game.py  (board and rules)  │
                 └──────────────────────────────┘
```

The key design decision is that **the agents never own the board**. A referee
holds the only copy, and an agent can touch it only through four tools:

| Tool | What it does |
|---|---|
| `get_board_state` | Shows the position and the moves so far |
| `get_legal_moves` | Lists every legal move |
| `analyze_position` | Reports facts: material, checks, undefended pieces |
| `make_move` | Plays a move and ends the turn |

A model that imagines an illegal move gets an error back and can try again; it
cannot corrupt the game.

## What it demonstrates

- **The agent loop** ([agent.py](chess_agents/agent.py)): call Claude, run the
  tools it asks for, send the results back, repeat until it moves.
- **Tool use**: tool schemas, tool results, error results and strict argument
  checking ([tools.py](chess_agents/tools.py)).
- **Orchestration** ([orchestrator.py](chess_agents/orchestrator.py)): turn
  taking, pause, single-step and stop.
- **Guardrails**: a tool-call budget and a time limit per turn, a limit on
  illegal move attempts, and a cap on game length. Breaking one forfeits the
  game. An API failure aborts the game instead, because it is not the player's
  fault.
- **Memory**: each agent leaves itself a short note when it moves and reads it
  back on its next turn. Turns are otherwise independent, which keeps the cost
  of a turn flat however long the game runs.
- **Distinct agents**: choose a model, a playing style and a thinking effort
  for each player, for example Opus against Haiku, or aggressive against
  positional.
- **Observability**: every thought, tool call, result and move is an event you
  can watch live, with token counts and an estimated cost per player.
- **Persistence**: games are saved after every move and can be resumed, with
  different models if you like.
- **Evaluation**: an optional script grades a finished game with Stockfish.

## Running it

You need Python 3.10 or newer and an Anthropic API key. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env      # then put your key in .env
.\.venv\Scripts\python.exe -m uvicorn chess_agents.server:app
```

Open <http://127.0.0.1:8000>, choose the players and press **Start game**.

Without an API key you can still try everything by choosing **Random mover
(no API)** for both players.

Each move is at least one Claude request, so a long game between two large
models uses a meaningful number of tokens. Start with a small move limit to see
what a game costs before running a full one.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The tests cover the rules wrapper, the agent loop (with a fake Claude client),
the referee's limits and controls, and the web server. None of them call the
real API.

## Grading a game with Stockfish (optional)

Install [Stockfish](https://stockfishchess.org), put it on your PATH or set
`STOCKFISH_PATH` in `.env`, then:

```powershell
.\.venv\Scripts\python.exe -m chess_agents.evaluate games\<game-id>.pgn
```

It prints each player's average centipawn loss and counts of inaccuracies,
mistakes and blunders.

## Project layout

| File | Purpose |
|---|---|
| [chess_agents/config.py](chess_agents/config.py) | Models, prices, playing styles, limits |
| [chess_agents/game.py](chess_agents/game.py) | The board and rules, built on python-chess |
| [chess_agents/tools.py](chess_agents/tools.py) | Tool definitions and read-only tool handlers |
| [chess_agents/agent.py](chess_agents/agent.py) | The Claude agent loop, and a random mover |
| [chess_agents/orchestrator.py](chess_agents/orchestrator.py) | The referee and game runner |
| [chess_agents/storage.py](chess_agents/storage.py) | Saving and loading games |
| [chess_agents/server.py](chess_agents/server.py) | Web server and event stream |
| [chess_agents/static/index.html](chess_agents/static/index.html) | The web page |
| [chess_agents/evaluate.py](chess_agents/evaluate.py) | Stockfish grading |
| [tests/](tests/) | Automated tests |
