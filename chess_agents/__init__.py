"""Two Claude agents playing chess against each other.

Module map (read in this order to follow the design):

    config.py        models, prices, personas and game limits
    game.py          the chess board and rules (wraps python-chess)
    tools.py         the tools an agent can call, as Claude tool schemas
    agent.py         one player: the Claude tool-use loop for a single turn
    orchestrator.py  the referee: turn order, budgets, events, pause/stop
    storage.py       saving and loading games on disk
    server.py        the web server that streams a game to the browser
    evaluate.py      optional Stockfish grading of a finished game
"""
