# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class IsinTest(nn.Module):
    """Tests torch.isin with int32 and int64 tensors of various sizes.

    Each call returns a 1D bool tensor indicating which elements of the
    first argument are present in the second argument (the set).
    """

    def forward(
        self,
        elements_i32_1k: Tensor,
        set_i32_1k: Tensor,
        elements_i64_1k: Tensor,
        set_i64_1k: Tensor,
        elements_i32_10k: Tensor,
        set_i32_10k: Tensor,
        elements_i64_100k: Tensor,
        set_i64_100k: Tensor,
        elements_i32_nz: Tensor,
        set_i32_with_zero: Tensor,
        elements_i64_nz: Tensor,
        set_i64_with_zero: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r1 = torch.isin(elements_i32_1k, set_i32_1k)
        r2 = torch.isin(elements_i64_1k, set_i64_1k)
        r3 = torch.isin(elements_i32_10k, set_i32_10k)
        r4 = torch.isin(elements_i64_100k, set_i64_100k)
        r5 = torch.isin(elements_i32_nz, set_i32_with_zero)
        r6 = torch.isin(elements_i64_nz, set_i64_with_zero)
        return r1, r2, r3, r4, r5, r6
