from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .games import Game
from .net import PolicyValueNet


@dataclass
class Edge:
    prior: float
    n: int = 0
    w: float = 0.0
    child: "Node | None" = None

    @property
    def q(self) -> float:
        return 0.0 if self.n == 0 else self.w / self.n


@dataclass
class Node:
    state: Game
    to_play: int
    edges: dict[int, Edge] = field(default_factory=dict)
    expanded: bool = False
    value: float = 0.0


class MCTS:
    def __init__(self, net: PolicyValueNet, sims: int = 64, c_puct: float = 1.5, dirichlet: float = 0.3, d_eps: float = 0.25):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet = dirichlet
        self.d_eps = d_eps
        self.net.eval()

    @torch.no_grad()
    def infer(self, state: Game) -> tuple[np.ndarray, float]:
        x = torch.from_numpy(state.encode()).unsqueeze(0)
        logits, value = self.net(x)
        n = state.spec.action_size
        legal = state.legal_moves()
        mask = torch.full((n,), -1e9)
        mask[legal] = 0
        prior = F.softmax(logits[0] + mask, dim=0).cpu().numpy()
        return prior, float(value[0])

    def search(self, state: Game, add_noise: bool = False) -> tuple[np.ndarray, Node]:
        root = Node(state=state.copy(), to_play=state.to_play)
        self._expand(root)
        if add_noise and root.edges:
            noise = np.random.dirichlet([self.dirichlet] * len(root.edges))
            for i, e in enumerate(root.edges.values()):
                e.prior = (1 - self.d_eps) * e.prior + self.d_eps * float(noise[i])
        for _ in range(self.sims):
            self._simulate(root)
        n = state.spec.action_size
        visits = np.zeros(n, dtype=np.float64)
        for a, e in root.edges.items():
            visits[a] = e.n
        if visits.sum() <= 0:
            legal = state.legal_moves()
            if legal:
                visits[legal] = 1.0
            else:
                visits[:] = 1.0
        pi = visits / visits.sum()
        return pi, root

    def _expand(self, node: Node) -> float:
        out = node.state.outcome()
        if out.terminal:
            node.expanded = True
            if out.winner == 0:
                node.value = 0.0
            else:
                node.value = 1.0 if out.winner == node.to_play else -1.0
            return node.value
        prior, value = self.infer(node.state)
        for a in node.state.legal_moves():
            node.edges[a] = Edge(prior=float(prior[a]))
        node.expanded = True
        node.value = value
        return value

    def _simulate(self, node: Node) -> float:
        out = node.state.outcome()
        if out.terminal:
            if out.winner == 0:
                return 0.0
            return 1.0 if out.winner == node.to_play else -1.0
        if not node.expanded:
            return self._expand(node)
        if not node.edges:
            return node.value
        action, edge = self._select(node)
        if edge.child is None:
            child_state = node.state.play(action)
            edge.child = Node(state=child_state, to_play=child_state.to_play)
        v_child = self._simulate(edge.child)
        v = -v_child
        edge.n += 1
        edge.w += v
        return v

    def _select(self, node: Node) -> tuple[int, Edge]:
        nsum = math.sqrt(sum(e.n for e in node.edges.values()) + 1e-8)
        best_a, best_e, best = -1, None, -1e9
        for a, e in node.edges.items():
            u = e.q + self.c_puct * e.prior * nsum / (1 + e.n)
            if u > best:
                best, best_a, best_e = u, a, e
        assert best_e is not None
        return best_a, best_e


def tree_view(root: Node, max_actions: int = 8) -> list[dict]:
    rows = []
    nsum = sum(e.n for e in root.edges.values()) or 1
    ranked = sorted(root.edges.items(), key=lambda kv: kv[1].n, reverse=True)[:max_actions]
    for a, e in ranked:
        rows.append(
            {
                "action": a,
                "col": a,
                "label": root.state.action_label(a),
                "n": e.n,
                "q": round(e.q, 4),
                "p": round(e.prior, 4),
                "share": round(e.n / nsum, 4),
            }
        )
    return rows


def principal_variation(root: Node, depth: int = 6) -> list[int]:
    pv = []
    node: Node | None = root
    for _ in range(depth):
        if node is None or not node.edges:
            break
        a, e = max(node.edges.items(), key=lambda kv: kv[1].n)
        pv.append(a)
        node = e.child
    return pv
