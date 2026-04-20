# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class CumsumTestPreproc(nn.Module):
    """Cumulative sum tests for different sizes and dtypes.

    Inputs: float_1k, float_10k, float_100k (float32),
            int_1k, int_10k, int_100k (int64).

    Returns cumsum(dim=0) for each input.
    """

    def forward(
        self,
        float_1k: Tensor,
        float_10k: Tensor,
        float_100k: Tensor,
        int_1k: Tensor,
        int_10k: Tensor,
        int_100k: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        return (
            torch.cumsum(float_1k, dim=0),
            torch.cumsum(float_10k, dim=0),
            torch.cumsum(float_100k, dim=0),
            torch.cumsum(int_1k, dim=0),
            torch.cumsum(int_10k, dim=0),
            torch.cumsum(int_100k, dim=0),
        )
