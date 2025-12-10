from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn

from ..utils.base import BaseModule
from ..utils.typing import *


class TimestepEmbedder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 0
        out_channels: int = 0

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.linear = nn.Linear(self.cfg.in_channels, self.cfg.out_channels)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, timesteps: Tensor) -> Tensor:
        timestep_encoding = self.timestep_embedding(timesteps, self.cfg.in_channels)

        embedding = self.linear(timestep_encoding)
        return embedding
