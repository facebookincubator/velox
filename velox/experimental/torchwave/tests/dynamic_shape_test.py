# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class DynamicShapeTestPreproc(nn.Module):
    """Tests data-dependent dynamic shapes: arange from sum, ones from length values.

    Inputs: lengths (20 int64 values between 1 and 4)
    Outputs: s (scalar sum), r (arange(s)), o (ones shaped by lengths[2] x lengths[3])
    """

    def forward(
        self,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        s = torch.sum(lengths)
        r = torch.arange(s.item())
        o = torch.ones(size=[int(lengths[2].item()), int(lengths[3].item())])
        return s, r, o
