from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .games import Spec


class Residual(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(x + h)


class PolicyValueNet(nn.Module):
    """AlphaZero dual head. Connect Four defaults match the 80-game checkpoint."""

    def __init__(self, spec: Spec | None = None, ch: int = 64, blocks: int = 4):
        super().__init__()
        if spec is None:
            from .games.connect4 import SPEC

            spec = SPEC
        self.spec = spec
        in_ch, rows, cols, actions = spec.in_ch, spec.rows, spec.cols, spec.action_size
        self.stem = nn.Sequential(nn.Conv2d(in_ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*[Residual(ch) for _ in range(blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(ch, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.p_fc = nn.Linear(16 * rows * cols, actions)
        self.v_conv = nn.Sequential(nn.Conv2d(ch, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.v_fc = nn.Sequential(nn.Linear(8 * rows * cols, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))
        logits = self.p_fc(self.p_conv(h).flatten(1))
        value = self.v_fc(self.v_conv(h).flatten(1)).squeeze(-1)
        return logits, value

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
