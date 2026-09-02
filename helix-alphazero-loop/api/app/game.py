from __future__ import annotations

"""Back-compat: Connect Four plus the multi-game registry."""

from .games import EMPTY, P1, P2, ConnectFour, catalog, make_game, planes_to_tensor
from .games.connect4 import SPEC

ROWS, COLS, WIN = SPEC.rows, SPEC.cols, 4

__all__ = [
    "COLS",
    "ConnectFour",
    "EMPTY",
    "P1",
    "P2",
    "ROWS",
    "WIN",
    "catalog",
    "make_game",
    "planes_to_tensor",
]
