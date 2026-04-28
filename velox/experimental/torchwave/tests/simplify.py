# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class SimplifyPreproc(nn.Module):
    """Tests simplification of redundant reshapes."""

    def forward(self, input1: Tensor) -> Tensor:
        return torch.reshape(torch.reshape(input1, [-1]), [-1])
