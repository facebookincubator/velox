# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from torch import nn, Tensor


class IndexSelectTest(nn.Module):
    """Tests fused elementwise index_select, standalone and broadcast.

    Exercises:
    1. Standalone index_select along each of a 3-D tensor's dims (5 indices).
    2. Broadcast: a 1-element index along each of dims 0, 1, 2 of three cubes,
       summed so each slice broadcasts over the other two dims.
    3. Mixed rank: a 3-D slice summed with a 2-D slice (exercises the
       right-alignment of a lower-rank operand within the expression output).
    4. Elementwise-produced source: index_select over the result of an
       elementwise add, as a standalone output. The source is read as a whole
       tensor, so the fused kernel materializes the add's result and then reads
       it by random access. That gives the all-elementwise op two memory-backed
       outputs (the add's result and index_select's result), which is allowed;
       a barrier orders the producer before the random-access read.
    5. Multi-op elementwise source: index_select over a two-op elementwise chain
       ((a + b) * c), exercising the same two-output path with a deeper border.
    """

    def forward(
        self,
        base: Tensor,
        idx0: Tensor,
        idx1: Tensor,
        idx2: Tensor,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        one0: Tensor,
        one1: Tensor,
        one2: Tensor,
        mat2d: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        s0 = torch.index_select(base, 0, idx0)
        s1 = torch.index_select(base, 1, idx1)
        s2 = torch.index_select(base, 2, idx2)
        cube = (
            torch.index_select(a, 0, one0)
            + torch.index_select(b, 1, one1)
            + torch.index_select(c, 2, one2)
        )
        mixed = torch.index_select(a, 0, one0) + torch.index_select(mat2d, 0, one0)
        ew_source = torch.index_select(a + b, 0, one0)
        ew_chain = torch.index_select((a + b) * c, 0, one0)
        return s0, s1, s2, cube, mixed, ew_source, ew_chain
