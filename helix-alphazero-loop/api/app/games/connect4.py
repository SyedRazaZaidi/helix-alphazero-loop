from __future__ import annotations

from .base import EMPTY, P1, P2, Game, Outcome, Spec, line_win

SPEC = Spec(
    id="connect4",
    name="Connect Four",
    rows=6,
    cols=7,
    action_size=7,
    in_ch=3,
    layout="drop",
    win="four-in-a-row, gravity drop",
    blurb="Classic 6×7. Actions are columns. Last trained checkpoint lives here.",
)


class ConnectFour(Game):
    spec = SPEC

    def legal_moves(self) -> list[int]:
        return [c for c in range(self.spec.cols) if self.board[0, c] == EMPTY]

    def play(self, action: int) -> ConnectFour:
        if action not in self.legal_moves():
            raise ValueError(f"illegal column {action}")
        nxt = self.copy()
        for r in range(self.spec.rows - 1, -1, -1):
            if nxt.board[r, action] == EMPTY:
                nxt.board[r, action] = nxt.to_play
                break
        nxt.to_play = P2 if nxt.to_play == P1 else P1
        return nxt

    def drop(self, col: int) -> ConnectFour:
        return self.play(col)

    def outcome(self) -> Outcome:
        if line_win(self.board, 4):
            winner = P2 if self.to_play == P1 else P1
            return Outcome(winner=winner, terminal=True)
        if not self.legal_moves():
            return Outcome(winner=0, terminal=True)
        return Outcome(winner=0, terminal=False)

    def action_label(self, action: int) -> str:
        return str(int(action))
