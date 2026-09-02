from __future__ import annotations

from collections import deque

import numpy as np

from .base import EMPTY, P1, P2, Game, Outcome, Spec

SPEC = Spec(
    id="hex",
    name="Hex 6×6",
    rows=6,
    cols=6,
    action_size=36,
    in_ch=3,
    layout="hex",
    win="P1 left–right · P2 top–bottom, 6-neighbor",
    blurb="Shannon switching game. Connection, not alignment. Hex is PSPACE-hard; 6×6 is the CPU cut.",
)

# odd-r horizontal offset
_EVEN = ((0, -1), (0, 1), (-1, -1), (-1, 0), (1, -1), (1, 0))
_ODD = ((0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1))


def _neighbors(r: int, c: int, rows: int, cols: int) -> list[tuple[int, int]]:
    dirs = _ODD if r % 2 else _EVEN
    out = []
    for dr, dc in dirs:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def _connected(board: np.ndarray, player: int, starts: list[tuple[int, int]], is_goal) -> bool:
    rows, cols = board.shape
    seen: set[tuple[int, int]] = set()
    q: deque[tuple[int, int]] = deque()
    for cell in starts:
        if board[cell] == player:
            q.append(cell)
            seen.add(cell)
    while q:
        r, c = q.popleft()
        if is_goal(r, c):
            return True
        for rr, cc in _neighbors(r, c, rows, cols):
            if (rr, cc) not in seen and board[rr, cc] == player:
                seen.add((rr, cc))
                q.append((rr, cc))
    return False


class Hex(Game):
    spec = SPEC

    def legal_moves(self) -> list[int]:
        rows, cols = self.spec.rows, self.spec.cols
        return [r * cols + c for r in range(rows) for c in range(cols) if self.board[r, c] == EMPTY]

    def play(self, action: int) -> Hex:
        if action not in self.legal_moves():
            raise ValueError(f"illegal cell {action}")
        nxt = self.copy()
        r, c = divmod(int(action), self.spec.cols)
        nxt.board[r, c] = nxt.to_play
        nxt.to_play = P2 if nxt.to_play == P1 else P1
        return nxt

    def outcome(self) -> Outcome:
        rows, cols = self.spec.rows, self.spec.cols
        p1_starts = [(r, 0) for r in range(rows)]
        p2_starts = [(0, c) for c in range(cols)]
        if _connected(self.board, P1, p1_starts, lambda r, c: c == cols - 1):
            return Outcome(winner=P1, terminal=True)
        if _connected(self.board, P2, p2_starts, lambda r, c: r == rows - 1):
            return Outcome(winner=P2, terminal=True)
        if not self.legal_moves():
            return Outcome(winner=0, terminal=True)
        return Outcome(winner=0, terminal=False)
