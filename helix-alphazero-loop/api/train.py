from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import torch
from torch.optim import Adam

from app.arena import vs_random
from app.games import make_game
from app.net import PolicyValueNet
from app.selfplay import play_game, train_step

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="connect4", choices=["connect4", "gomoku", "hex", "othello"])
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sims", type=int, default=24)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=20)
    args = p.parse_args()
    ART.mkdir(exist_ok=True)
    random.seed(7)
    torch.manual_seed(7)
    spec = make_game(args.game).spec
    net = PolicyValueNet(spec)
    opt = Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buf: deque = deque(maxlen=args.buffer)
    ckpt = ART / f"helix-{args.game}.pt"
    metrics = ART / f"metrics-{args.game}.jsonl"
    with metrics.open("w", encoding="utf-8") as log:
        for g in range(1, args.games + 1):
            temp = 1.0 if g < args.games * 0.6 else 0.3
            samples = play_game(net, sims=args.sims, temperature=temp, game_id=args.game)
            buf.extend(samples)
            stats = {"game": g, "env": args.game, "buffer": len(buf), "moves": len(samples)}
            if len(buf) >= min(args.batch, 16) and buf:
                take = min(args.batch, len(buf))
                batch = random.sample(list(buf), take)
                stats.update(train_step(net, batch, opt))
            if g % args.eval_every == 0 or g == args.games:
                ev = vs_random(net, sims=max(12, args.sims // 2), games=8, game_id=args.game)
                stats["vs_random"] = ev
                torch.save({"state": net.state_dict(), "games": g, "params": net.n_params(), "env": args.game}, ckpt)
            log.write(json.dumps(stats) + "\n")
            log.flush()
            print(json.dumps(stats), flush=True)
    print(f"wrote {ckpt}", flush=True)


if __name__ == "__main__":
    main()
