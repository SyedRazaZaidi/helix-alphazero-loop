from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .arena import vs_random
from .games import catalog, make_game, spec as game_spec
from .mcts import MCTS, principal_variation, tree_view
from .net import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def ckpt_path(game_id: str) -> Path:
    named = ART / f"helix-{game_id}.pt"
    if named.exists():
        return named
    if game_id == "connect4":
        legacy = ART / "helix.pt"
        if legacy.exists():
            return legacy
    return named


def metrics_path(game_id: str) -> Path:
    named = ART / f"metrics-{game_id}.jsonl"
    if named.exists():
        return named
    if game_id == "connect4":
        legacy = ART / "metrics.jsonl"
        if legacy.exists():
            return legacy
    return named


@lru_cache(maxsize=8)
def load_net(game_id: str = "connect4") -> PolicyValueNet:
    s = game_spec(game_id)
    net = PolicyValueNet(s)
    path = ckpt_path(game_id)
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(blob["state"])
    net.eval()
    return net


def warmup() -> None:
    for row in catalog():
        load_net(row["id"])


def checkpoint_info(game_id: str = "connect4") -> dict:
    s = game_spec(game_id)
    path = ckpt_path(game_id)
    net = load_net(game_id)
    games = 0
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        games = int(blob.get("games", 0))
    return {
        "model": f"helix-pv-{s.id}",
        "game": s.id,
        "n_params": net.n_params(),
        "checkpoint_games": games,
        "has_checkpoint": path.exists(),
        "board": f"{s.name} {s.rows}x{s.cols}",
        "layout": s.layout,
        "win": s.win,
        "blurb": s.blurb,
        "games": catalog(),
        "note": "One AlphaZero loop, four rulesets: Connect Four, Gomoku, Hex, Othello. CPU. Not a chatbot.",
    }


def metrics_tail(game_id: str = "connect4", n: int = 80) -> list[dict]:
    path = metrics_path(game_id)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def think(grid: list[list[int]], to_play: int, sims: int = 64, mode: str = "mcts", game_id: str = "connect4") -> dict:
    net = load_net(game_id)
    state = make_game(game_id, grid, to_play)
    n = state.spec.action_size
    out = state.outcome()
    if out.terminal:
        snap = state.snapshot()
        snap.update(
            {
                "policy": [0.0] * n,
                "value": 0.0,
                "tree": [],
                "pv": [],
                "pv_labels": [],
                "choice": None,
                "mode": mode,
            }
        )
        return snap
    legal = state.legal_moves()
    if mode == "random":
        choice = int(np.random.choice(legal))
        prior = np.zeros(n)
        prior[legal] = 1 / len(legal)
        tree, pv, value = [], [], 0.0
        policy = [round(float(x), 4) for x in prior]
    elif mode == "net":
        prior, value = MCTS(net, sims=1).infer(state)
        choice = int(max(legal, key=lambda c: prior[c]))
        policy = [round(float(x), 4) for x in prior]
        tree = [
            {
                "action": i,
                "col": i,
                "label": state.action_label(i),
                "n": 0,
                "q": 0.0,
                "p": round(float(prior[i]), 4),
                "share": round(float(prior[i]), 4),
            }
            for i in legal[:12]
        ]
        pv = []
    else:
        mcts = MCTS(net, sims=max(8, int(sims)))
        pi, root = mcts.search(state, add_noise=False)
        choice = int(pi.argmax())
        policy = [round(float(x), 4) for x in pi]
        tree = tree_view(root)
        pv = principal_variation(root)
        value = root.value
    snap = state.snapshot()
    snap.update(
        {
            "policy": policy,
            "value": round(float(value), 4),
            "tree": tree,
            "pv": pv,
            "pv_labels": [state.action_label(a) for a in pv],
            "choice": choice,
            "mode": mode,
            "sims": sims if mode == "mcts" else 0,
        }
    )
    return snap


def apply_move(grid: list[list[int]], to_play: int, action: int, game_id: str = "connect4") -> dict:
    state = make_game(game_id, grid, to_play).play(action)
    return state.snapshot()


def empty_board(game_id: str = "connect4") -> dict:
    return make_game(game_id).snapshot()


def quick_eval(game_id: str = "connect4") -> dict:
    net = load_net(game_id)
    return vs_random(net, sims=24, games=8, game_id=game_id)
