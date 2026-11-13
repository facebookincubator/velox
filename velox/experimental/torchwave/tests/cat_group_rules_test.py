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


class CatGroupRulesTest(nn.Module):
    """Concats of three operands, each in a form the allocation group has to
    take without falling back to copies inside the concat's own kernel.

    Every cat or stack of more than two operands gets a concat allocation
    group. Each operand either writes its band itself, is computed into it by
    the concat's kernel, or is moved there by a copy op of its own, side by
    side with the other copies. None is left to the concat's kernel to copy.

    Each repeat_interleave result is used twice, so it is materialized by a
    launch of its own, which is what lets the group hand it its band.

    Inputs:
        d0..d5 (64 longs each) and reps (64 longs) -- six 1-D gathers
    Outputs:
        inner: cat of three gathers
        outer: cat of 'inner' and two gathers. 'inner' is the result of a
               grouped concat, which its own group already owns, so it is
               copied into 'outer'.
        eager: cat around torch.unique, an eager op whose size is known only
               after it runs, so its operand is copied
        readEager: 1 + eager, a reader of the result, which has to run after
               the copies
        top: the same form as 'eager', returned without a reader
        stacked: stack of two gathers and a sort, another eager op
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r0 = torch.repeat_interleave(d0, reps)
        r1 = torch.repeat_interleave(d1, reps)
        r2 = torch.repeat_interleave(d2, reps)
        r3 = torch.repeat_interleave(d3, reps)
        r4 = torch.repeat_interleave(d4, reps)
        r5 = torch.repeat_interleave(d5, reps)

        inner = torch.cat([r0, r1, r2])
        outer = torch.cat([inner, r3, r4])

        eager = torch.cat([r5, torch.unique(d5), r0])
        read_eager = 1 + eager

        top = torch.cat([r3, torch.unique(d4), r4])

        stacked = torch.stack([r0, r1, torch.sort(r2).values])

        marks = r0 + r1 + r2 + r3 + r4 + r5

        return inner, outer, eager, read_eager, top, stacked, marks
