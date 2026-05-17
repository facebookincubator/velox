# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class ElementTestPreproc(nn.Module):
    """Simple element-wise preproc for testing sigmoid packaging.

    Inputs: a, b, c (10000 longs each), d, e, f (100000 longs each)
    Outputs: o1 = a + b - c, o2 = c + 3*b - a, o3 = c - a, o4 = d + 2*e, o5 = f - e + d
    """

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
        e: Tensor,
        f: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        o1 = a + b - c
        o2 = torch.add(c, b, alpha=3) - a
        o3 = c - a
        o4 = torch.add(d, e, alpha=2)
        o5 = f - e + d
        return o1, o2, o3, o4, o5
