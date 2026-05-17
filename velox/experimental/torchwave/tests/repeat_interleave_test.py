# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class RepeatInterleaveTestPreproc(nn.Module):
    """repeat_interleave test with variable tensor repeats.

    Inputs:
        data (1000 longs) - values to repeat
        repeats (1000 longs) - repeat counts per element, values 0-3

    Output: repeat_interleave(data, repeats) - variable repeats per element
    """

    def forward(
        self,
        data: Tensor,
        repeats: Tensor,
    ) -> Tensor:
        return torch.repeat_interleave(data, repeats)
