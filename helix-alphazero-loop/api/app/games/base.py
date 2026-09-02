from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

EMPTY, P1, P2 = 0, 1, 2


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    rows: int
    cols: int
    action_size: int
    in_ch: int
    layout: str  # drop | cells | hex
    win: str
    blurb: str


@dataclass
class Outcome:
    winner: int  # 0 draw, 1/2 player
    terminal: bool


class Game:
    spec: Spec

    def __init__(self, board: np.ndarray | None = None, to_play: int = P1):
        self.board = self._empty() if board is None else board.astype(np.int8, copy=True)
        self.to_play = int(to_play)

    def _empty(self) -> np.ndarray:
        return np.zeros((self.spec.rows, self.spec.cols), dtype=np.int8)

    def copy(self) -> Game:
        return self.__class__(self.board.copy(), self.to_play)

    def legal_moves(self) -> list[int]:
        raise NotImplementedError

    def play(self, action: int) -> Game:
        raise NotImplementedError

    def outcome(self) -> Outcome:
        raise NotImplementedError

    def encode(self) -> np.ndarray:
        me = self.to_play
        opp = P2 if me == P1 else P1
        x = np.zeros((self.spec.in_ch, self.spec.rows, self.spec.cols), dtype=np.float32)
        x[0] = self.board == me
        x[1] = self.board == opp
        x[2] = 1.0 if me == P1 else 0.0
        return x

    def grid(self) -> list[list[int]]:
        return self.board.tolist()

    def action_label(self, action: int) -> str:
        r, c = divmod(int(action), self.spec.cols)
        return f"{r},{c}"

    def snapshot(self) -> dict:
        out = self.outcome()
        return {
            "game": self.spec.id,
            "name": self.spec.name,
            "layout": self.spec.layout,
            "rows": self.spec.rows,
            "cols": self.spec.cols,
            "grid": self.grid(),
            "to_play": self.to_play,
            "terminal": out.terminal,
            "winner": out.winner,
            "legal": self.legal_moves(),
            "blurb": self.spec.blurb,
            "win": self.spec.win,
        }


def planes_to_tensor(batch: Iterable[np.ndarray]):
    import torch

    return torch.from_numpy(np.stack(list(batch)))


def line_win(board: np.ndarray, n: int) -> bool:
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            p = int(board[r, c])
            if p == EMPTY:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                ok = True
                for k in range(n):
                    rr, cc = r + dr * k, c + dc * k
                    if not (0 <= rr < rows and 0 <= cc < cols) or int(board[rr, cc]) != p:
                        ok = False
                        break
                if ok:
                    return True
    return False
