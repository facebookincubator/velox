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


class CatAllocGroupTest(nn.Module):
    """Wide concats over gathers that are materialized before the concat runs.

    Stands in for the final concat of the ROO preproc graph. An operand the
    concat's own kernel computes already writes straight into the result -- the
    concat hands it a view of the region it occupies -- so nothing is saved
    there. What costs an allocation and a copy is an operand that arrives from
    the far side of a kernel boundary: it has a buffer of its own and
    __concatCopy moves it in. A concat allocation group places the result at the
    step that produces those operands and gives each of them its region instead.

    Each repeat_interleave result is used twice, so it is materialized on its
    own rather than folded into the concat's kernel -- the shape the ROO gathers
    have, where the same gather feeds the final concat and something else.

    Inputs:
        d0..d5 (64 longs each) and reps (64 longs) -- six 1-D gathers
        plain (32 longs) -- an operand the graph is handed rather than computes
        m0..m2 (4x8 floats) and mreps (4 longs) -- three 2-D gathers
        e0..e2 (64 longs each) -- gathers read only by the scaling below
    Outputs:
        wide:  cat of four gathers -- the case every operand is placed in
        mixed: cat of two gathers around a graph input, which has to be copied
               in while the operand after it still lands at the right offset
        pair:  cat of two -- below the threshold, so the ordinary path
        nd:    cat of three 2-D gathers along dim 0, where an operand's region
               of the result spans whole rows
        scaled: cat of three gathers with an elementwise op in between. Like
               'nd' the extents are behind the gathers' reserve functions, but
               the elementwise op has a size expression of its own, so only a
               walk up the producer chain shows the layout cannot be computed
               ahead of the operands.
        marks: the second use of every gather
    """

    def forward(
        self,
        d0: Tensor,
        d1: Tensor,
        d2: Tensor,
        d3: Tensor,
        d4: Tensor,
        d5: Tensor,
        reps: Tensor,
        plain: Tensor,
        m0: Tensor,
        m1: Tensor,
        m2: Tensor,
        mreps: Tensor,
        e0: Tensor,
        e1: Tensor,
        e2: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r0 = torch.repeat_interleave(d0, reps)
        r1 = torch.repeat_interleave(d1, reps)
        r2 = torch.repeat_interleave(d2, reps)
        r3 = torch.repeat_interleave(d3, reps)
        r4 = torch.repeat_interleave(d4, reps)
        r5 = torch.repeat_interleave(d5, reps)

        wide = torch.cat([r0, r1, r2, r3])
        mixed = torch.cat([r4, plain, r5])
        pair = torch.cat([d0 * 2, d0 + 1])

        n0 = torch.repeat_interleave(m0, mreps, dim=0)
        n1 = torch.repeat_interleave(m1, mreps, dim=0)
        n2 = torch.repeat_interleave(m2, mreps, dim=0)
        nd = torch.cat([n0, n1, n2], dim=0)

        scaled = torch.cat(
            [
                torch.repeat_interleave(e0, reps) * 3,
                torch.repeat_interleave(e1, reps) * 3,
                torch.repeat_interleave(e2, reps) * 3,
            ]
        )

        marks = r0 + r1 + r2 + r3 + r4 + r5

        return wide, mixed, pair, nd, scaled, marks
