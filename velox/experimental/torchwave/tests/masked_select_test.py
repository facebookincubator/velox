# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class MaskedSelectTestPreproc(nn.Module):
    """Masked select preproc for testing.

    Inputs: a, b (10000 longs each), where a[i] = i % 200, b[i] = i % 100.
    Output: torch.masked_select(a + b, a % 10 < 5)
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.masked_select(a + b, a % 10 < 5)
