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


class StackNdTest(nn.Module):
    """Stacks of 1-D, 2-D and 3-D operands along a new dim at every position.

    Each operand occupies a single position along the new dim, which is a
    strided slice of the result unless the new dim is the outermost one. o6
    reaches rank 4, the widest shape the kernel tensor descriptor holds.

    Inputs: x, y (7), a, b (6x5), d, e (2x3x4), all float.
    Outputs:
        o1: stack([x, y], dim=0)          -> 2x7
        o2: stack([x, y, x+y], dim=1)     -> 7x3
        o3: stack([a, b], dim=0)          -> 2x6x5
        o4: stack([a, b*2], dim=1)        -> 6x2x5
        o5: stack([a, b], dim=-1)         -> 6x5x2
        o6: stack([d, e+1], dim=3)        -> 2x3x4x2
    """

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        a: Tensor,
        b: Tensor,
        d: Tensor,
        e: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        o1 = torch.stack([x, y], dim=0)
        o2 = torch.stack([x, y, x + y], dim=1)
        o3 = torch.stack([a, b], dim=0)
        o4 = torch.stack([a, b * 2.0], dim=1)
        o5 = torch.stack([a, b], dim=-1)
        o6 = torch.stack([d, e + 1.0], dim=3)
        return o1, o2, o3, o4, o5, o6
