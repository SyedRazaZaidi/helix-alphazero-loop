from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .games import P1, P2, Game, make_game, planes_to_tensor
from .mcts import MCTS
from .net import PolicyValueNet


def play_game(net: PolicyValueNet, sims: int, temperature: float = 1.0, game_id: str = "connect4") -> list[tuple[np.ndarray, np.ndarray, int]]:
    mcts = MCTS(net, sims=sims)
    state: Game = make_game(game_id)
    n = state.spec.action_size
    trace: list[tuple[np.ndarray, np.ndarray, int]] = []
    while True:
        out = state.outcome()
        if out.terminal:
            z_map = {0: 0.0, P1: 1.0, P2: -1.0}
            z_p1 = z_map[out.winner]
            samples = []
            for planes, pi, player in trace:
                z = z_p1 if player == P1 else -z_p1
                samples.append((planes, pi, z))
            return samples
        add_noise = len(trace) < 8
        pi, _ = mcts.search(state, add_noise=add_noise)
        if temperature < 1e-3:
            col = int(np.argmax(pi))
        else:
            p = pi ** (1 / temperature)
            p = p / p.sum()
            col = int(np.random.choice(n, p=p))
            if col not in state.legal_moves():
                col = random.choice(state.legal_moves())
        trace.append((state.encode(), pi.astype(np.float32), state.to_play))
        state = state.play(col)


def train_step(net: PolicyValueNet, batch: list[tuple[np.ndarray, np.ndarray, float]], opt: Adam) -> dict:
    x = planes_to_tensor([b[0] for b in batch])
    pi = torch.from_numpy(np.stack([b[1] for b in batch]))
    z = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    net.train()
    logits, v = net(x)
    policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z)
    loss = policy_loss + value_loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
    opt.step()
    net.eval()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
    }
