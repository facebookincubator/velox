# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class ViewTestPreproc(nn.Module):
    """View test: masked_select followed by truncation and reshape.

    Inputs: a, b (10000 longs each), where a[i] = i % 200, b[i] = i % 100.
    Output: masked_select result truncated to a multiple of 4, then viewed as [-1, 4].
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        selected = torch.masked_select(a + b, a % 10 < 5)
        n = selected.shape[0]
        truncated = n - n % 4
        result = selected[:truncated]
        return result.view(-1, 4)
