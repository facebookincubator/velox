# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class ElementTest2Preproc(nn.Module):
    """Tests for allocating/casting elementwise ops: arange, zeros, to.dtype.

    Each op is tested as both a direct return value and as an operand
    to an elementwise op.

    Inputs: a, b (10000 int64 each), c, d (10000 float32 each).
    """

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        # arange.default: return value and as operand to add.
        ar = torch.arange(10000)
        ar_plus_a = torch.arange(10000) + a

        # arange.start: return value and as operand to sub.
        ar_s = torch.arange(5, 10005)
        ar_s_minus_b = torch.arange(5, 10005) - b

        # zeros (float32): return value and as operand to add.
        z = torch.zeros([10000])
        z_plus_c = torch.zeros([10000]) + c

        # to.dtype int->float: return value and as operand to add.
        a_float = a.to(torch.float32)
        cast_sum = a.to(torch.float32) + c

        # to.dtype float->double: return value and as operand to add.
        c_double = c.to(torch.float64)
        c_double_sum = c.to(torch.float64) + d.to(torch.float64)

        return (
            ar,
            ar_plus_a,
            ar_s,
            ar_s_minus_b,
            z,
            z_plus_c,
            a_float,
            cast_sum,
            c_double,
            c_double_sum,
        )
