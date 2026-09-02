from __future__ import annotations

import numpy as np

from .base import EMPTY, P1, P2, Game, Outcome, Spec

SPEC = Spec(
    id="othello",
    name="Othello 6×6",
    rows=6,
    cols=6,
    action_size=37,  # 36 cells + pass
    in_ch=3,
    layout="cells",
    win="disc count after flips · pass if no capture",
    blurb="Reversi. A move must sandwich opponent discs. Pass is action 36 when you have no capture.",
)

PASS = 36
DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


class Othello(Game):
    spec = SPEC

    def _empty(self) -> np.ndarray:
        b = np.zeros((6, 6), dtype=np.int8)
        b[2, 2] = P2
        b[2, 3] = P1
        b[3, 2] = P1
        b[3, 3] = P2
        return b

    def _flips(self, r: int, c: int, player: int) -> list[tuple[int, int]]:
        if self.board[r, c] != EMPTY:
            return []
        opp = P2 if player == P1 else P1
        taken: list[tuple[int, int]] = []
        for dr, dc in DIRS:
            line: list[tuple[int, int]] = []
            rr, cc = r + dr, c + dc
            while 0 <= rr < 6 and 0 <= cc < 6 and self.board[rr, cc] == opp:
                line.append((rr, cc))
                rr += dr
                cc += dc
            if line and 0 <= rr < 6 and 0 <= cc < 6 and self.board[rr, cc] == player:
                taken.extend(line)
        return taken

    def _placements(self, player: int) -> list[int]:
        out = []
        for r in range(6):
            for c in range(6):
                if self._flips(r, c, player):
                    out.append(r * 6 + c)
        return out

    def legal_moves(self) -> list[int]:
        place = self._placements(self.to_play)
        if place:
            return place
        if self._placements(P2 if self.to_play == P1 else P1):
            return [PASS]
        return []

    def play(self, action: int) -> Othello:
        legal = self.legal_moves()
        if action not in legal:
            raise ValueError(f"illegal action {action}")
        nxt = self.copy()
        if action == PASS:
            nxt.to_play = P2 if nxt.to_play == P1 else P1
            return nxt
        r, c = divmod(int(action), 6)
        for rr, cc in self._flips(r, c, self.to_play):
            nxt.board[rr, cc] = self.to_play
        nxt.board[r, c] = self.to_play
        nxt.to_play = P2 if nxt.to_play == P1 else P1
        return nxt

    def outcome(self) -> Outcome:
        if self.legal_moves():
            return Outcome(winner=0, terminal=False)
        n1 = int((self.board == P1).sum())
        n2 = int((self.board == P2).sum())
        if n1 > n2:
            return Outcome(winner=P1, terminal=True)
        if n2 > n1:
            return Outcome(winner=P2, terminal=True)
        return Outcome(winner=0, terminal=True)

    def action_label(self, action: int) -> str:
        if int(action) == PASS:
            return "pass"
        return super().action_label(action)
