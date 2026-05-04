# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class ExclusiveSumTestPreproc(nn.Module):
    """Exclusive sum tests for different sizes and dtypes.

    Inputs: float_1k, float_10k, float_100k (float32),
            int_1k, int_10k, int_100k (int64).

    Returns cat(zeros(1), cumsum(dim=0)) for each input, which
    torchwave rewrites to exclusive_sum.
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
            torch.cat([torch.zeros(1, dtype=float_1k.dtype), torch.cumsum(float_1k, dim=0)]),
            torch.cat([torch.zeros(1, dtype=float_10k.dtype), torch.cumsum(float_10k, dim=0)]),
            torch.cat([torch.zeros(1, dtype=float_100k.dtype), torch.cumsum(float_100k, dim=0)]),
            torch.cat([torch.zeros(1, dtype=int_1k.dtype), torch.cumsum(int_1k, dim=0)]),
            torch.cat([torch.zeros(1, dtype=int_10k.dtype), torch.cumsum(int_10k, dim=0)]),
            torch.cat([torch.zeros(1, dtype=int_100k.dtype), torch.cumsum(int_100k, dim=0)]),
        )
