from __future__ import annotations

from typing import Type

import numpy as np

from .base import EMPTY, P1, P2, Game, Outcome, Spec, planes_to_tensor
from .connect4 import ConnectFour
from .gomoku import Gomoku
from .hex import Hex
from .othello import Othello

FACTORIES: dict[str, Type[Game]] = {
    "connect4": ConnectFour,
    "gomoku": Gomoku,
    "hex": Hex,
    "othello": Othello,
}


def spec(game_id: str) -> Spec:
    return make_game(game_id).spec


def make_game(game_id: str, board: list[list[int]] | np.ndarray | None = None, to_play: int = P1) -> Game:
    cls = FACTORIES.get(game_id)
    if cls is None:
        raise ValueError(f"unknown game {game_id}. choose: {', '.join(FACTORIES)}")
    arr = None if board is None else np.array(board, dtype=np.int8)
    if arr is not None:
        s = cls.spec
        if arr.shape != (s.rows, s.cols):
            raise ValueError(f"{s.id} grid must be {s.rows}x{s.cols}")
    return cls(arr, to_play=to_play)


def catalog() -> list[dict]:
    out = []
    for gid in FACTORIES:
        s = FACTORIES[gid].spec
        out.append(
            {
                "id": s.id,
                "name": s.name,
                "rows": s.rows,
                "cols": s.cols,
                "action_size": s.action_size,
                "layout": s.layout,
                "win": s.win,
                "blurb": s.blurb,
            }
        )
    return out


__all__ = [
    "EMPTY",
    "P1",
    "P2",
    "ConnectFour",
    "FACTORIES",
    "Game",
    "Gomoku",
    "Hex",
    "Othello",
    "Outcome",
    "Spec",
    "catalog",
    "make_game",
    "planes_to_tensor",
    "spec",
]
