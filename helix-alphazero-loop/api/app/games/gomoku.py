from __future__ import annotations

from .base import EMPTY, P1, P2, Game, Outcome, Spec, line_win

SPEC = Spec(
    id="gomoku",
    name="Gomoku 8×8",
    rows=8,
    cols=8,
    action_size=64,
    in_ch=3,
    layout="cells",
    win="five-in-a-row, free placement",
    blurb="Same family as AlphaGo’s board games: place a stone, make five. 8×8 so it fits CPU MCTS.",
)


class Gomoku(Game):
    spec = SPEC

    def legal_moves(self) -> list[int]:
        rows, cols = self.spec.rows, self.spec.cols
        return [r * cols + c for r in range(rows) for c in range(cols) if self.board[r, c] == EMPTY]

    def play(self, action: int) -> Gomoku:
        if action not in self.legal_moves():
            raise ValueError(f"illegal cell {action}")
        nxt = self.copy()
        r, c = divmod(int(action), self.spec.cols)
        nxt.board[r, c] = nxt.to_play
        nxt.to_play = P2 if nxt.to_play == P1 else P1
        return nxt

    def outcome(self) -> Outcome:
        if line_win(self.board, 5):
            winner = P2 if self.to_play == P1 else P1
            return Outcome(winner=winner, terminal=True)
        if not self.legal_moves():
            return Outcome(winner=0, terminal=True)
        return Outcome(winner=0, terminal=False)
