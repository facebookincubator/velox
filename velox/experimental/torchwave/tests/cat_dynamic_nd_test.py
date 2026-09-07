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


class CatDynamicNdTest(nn.Module):
    """Cats whose operand length is decided on device, at rank 1 and rank 2.

    The 1-D cat absorbs its masked_select: the same kernel patches the
    following operands' view bases once the length is known on device. A wider
    cat cannot -- the host allocates the result and hands each operand a
    (strided) view of it, which needs every shape up front -- so its
    masked_select ends its kernel first and the length is read back before the
    cat's launch sizes the result.

    Inputs: x (200), mask (200, bool), pad (3x1), ipad (2x1, int64).
    Outputs:
        o1: cat([masked_select(x, mask), x + 1])         -> 1-D
        o2: cat([pad, column], dim=0)                    -> (3+k)x1
        o3: cat([column, column*3], dim=1)               -> kx2
        o4: cat([ipad, nonzero(mask)], dim=0)            -> (2+k)x1
    """

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        pad: Tensor,
        ipad: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        flat = torch.masked_select(x, mask)
        o1 = torch.cat([flat, x + 1.0])

        # A separate masked_select so the 2-D path's kernel boundary cannot be
        # confused with one the 1-D cat forced.
        sel = torch.masked_select(x * 2.0, mask)
        column = sel.unsqueeze(1)
        o2 = torch.cat([pad, column], dim=0)
        o3 = torch.cat([column, column * 3.0], dim=1)

        # nonzero already returns (k, 1), so its device-set row count reaches
        # the cat with no reshaping view in between. In single-block and
        # cooperative-grid mode nothing else would end its kernel, so this is
        # the case that needs the concat's own operand pushdown.
        o4 = torch.cat([ipad, torch.nonzero(mask)], dim=0)
        return o1, o2, o3, o4
