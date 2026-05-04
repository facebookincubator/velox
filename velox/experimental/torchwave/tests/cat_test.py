# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class CatTestPreproc(nn.Module):
    """Cat preproc for testing concatenation with various 1D inputs.

    Inputs: a, b (10000 longs each), where a[i] = i % 200, b[i] = i % 100.
    Outputs:
        o1: cat of zeros, ones, arange, and elementwise results
        o2: cat of cats (nested), testing flatten
        o3: cat of multiple masked_selects
    """

    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z = torch.zeros(1000, dtype=a.dtype)
        o = torch.ones(2000, dtype=a.dtype)
        r = torch.arange(0, 3000, dtype=a.dtype)
        s = a + b
        d = a - b
        p = a + a

        # o1: simple cat of various 1D inputs
        o1 = torch.cat([z, o, r, s, d])

        # Build nested cats to test flatten
        cat_inner1 = torch.cat([z, r])
        cat_inner2 = torch.cat([s, p])
        # o2: cat of cats plus a plain tensor
        o2 = torch.cat([cat_inner1, o, cat_inner2, d])

        # cumsum input for an additional cat element
        cs = torch.cumsum(a, dim=0)
        cat_inner3 = torch.cat([cs, d])
        o2b = torch.cat([cat_inner3, r])

        # o3: cat of multiple masked_selects using tensor-only comparisons
        ms1 = torch.masked_select(s, s + s > s)
        ms2 = torch.masked_select(p, d > a)
        ms3 = torch.masked_select(d, b > a)
        o3 = torch.cat([ms1, ms2, ms3])

        # Combine o2 and o2b for the second output
        o2_final = torch.cat([o2, o2b])

        return o1, o2_final, o3
