from __future__ import annotations

import random

from .games import P1, P2, Game, make_game
from .mcts import MCTS
from .net import PolicyValueNet


def play_match(a: PolicyValueNet, b: PolicyValueNet, sims: int, games: int, game_id: str = "connect4") -> dict:
    wa = wb = dr = 0
    for i in range(games):
        first, second = (a, b) if i % 2 == 0 else (b, a)
        winner = _one_game(first, second, sims, game_id)
        if winner == 0:
            dr += 1
        elif (winner == P1 and i % 2 == 0) or (winner == P2 and i % 2 == 1):
            wa += 1
        else:
            wb += 1
    return {"a": wa, "b": wb, "draw": dr, "games": games}


def vs_random(net: PolicyValueNet, sims: int, games: int = 12, game_id: str = "connect4") -> dict:
    wins = draws = 0
    for i in range(games):
        net_is_p1 = i % 2 == 0
        winner = _net_vs_random(net, sims, net_is_p1, game_id)
        if winner == 0:
            draws += 1
        elif (winner == P1 and net_is_p1) or (winner == P2 and not net_is_p1):
            wins += 1
    return {"wins": wins, "draws": draws, "losses": games - wins - draws, "games": games}


def _one_game(p1_net: PolicyValueNet, p2_net: PolicyValueNet, sims: int, game_id: str) -> int:
    s: Game = make_game(game_id)
    m1, m2 = MCTS(p1_net, sims=sims), MCTS(p2_net, sims=sims)
    while True:
        out = s.outcome()
        if out.terminal:
            return out.winner
        mcts = m1 if s.to_play == P1 else m2
        pi, _ = mcts.search(s, add_noise=False)
        s = s.play(int(pi.argmax()))


def _net_vs_random(net: PolicyValueNet, sims: int, net_is_p1: bool, game_id: str) -> int:
    s: Game = make_game(game_id)
    mcts = MCTS(net, sims=sims)
    while True:
        out = s.outcome()
        if out.terminal:
            return out.winner
        net_turn = (s.to_play == P1) == net_is_p1
        if net_turn:
            pi, _ = mcts.search(s, add_noise=False)
            col = int(pi.argmax())
        else:
            col = random.choice(s.legal_moves())
        s = s.play(col)
