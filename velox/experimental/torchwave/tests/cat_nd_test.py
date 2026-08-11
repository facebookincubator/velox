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


class CatNdTest(nn.Module):
    """Cats of 2-D and 3-D tensors along every dimension.

    On dim 0 each operand's region of the result is contiguous; on any other
    dim it is a strided band. Both plain inputs (which the kernel copies) and
    computed operands (which write their result straight into their band) are
    covered.

    Inputs: a (6x5), b (4x5), c (6x3), d (2x3x4), e (2x3x4), all float.
    Outputs:
        o1: cat([a, b], dim=0)               -> 10x5
        o2: cat([a, c], dim=1)               -> 6x8
        o3: cat([a*2, c+1, a-0.5], dim=-1)   -> 6x13
        o4: cat([d, e], dim=0)               -> 4x3x4
        o5: cat([d, e*3], dim=1)             -> 2x6x4
        o6: cat([d, e, d+e], dim=2)          -> 2x3x12
    """

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
        e: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        o1 = torch.cat([a, b], dim=0)
        o2 = torch.cat([a, c], dim=1)
        o3 = torch.cat([a * 2.0, c + 1.0, a - 0.5], dim=-1)
        o4 = torch.cat([d, e], dim=0)
        o5 = torch.cat([d, e * 3.0], dim=1)
        o6 = torch.cat([d, e, d + e], dim=2)
        return o1, o2, o3, o4, o5, o6
