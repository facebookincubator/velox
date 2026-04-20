# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch import nn, Tensor


class ElementShapeTest(nn.Module):
    """Elementwise ops on tensors with broadcast-compatible shapes.

    Tests broadcast correctness across shapes:
    [10, 9, 8], [1, 9, 8], [10, 1, 8], [10, 9, 1], [1, 1, 1], []
    """

    def forward(
        self,
        full: Tensor,
        bc_dim0: Tensor,
        bc_dim1: Tensor,
        bc_dim2: Tensor,
        bc_all: Tensor,
        scalar: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        o1 = full + bc_dim0
        o2 = full * bc_dim1
        o3 = full - bc_dim2
        o4 = full + bc_all
        o5 = full + scalar
        return o1, o2, o3, o4, o5
