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


class CatParallelFillTest(nn.Module):
    """A wide concat no operand of which writes its own band, and a reader.

    The case cat_alloc_group_test does not reach: there every wide cat has at
    most one operand the allocation group cannot carve, so the copies never have
    to be looked at as a set. Here the graph hands over all six operands of
    'handed', so no launch writes any of them and the group carves none. Each
    operand gets a copy op of its own, and the point of the test is that all six
    run in the concat's own step -- side by side -- rather than as a chain of
    __concatCopy calls walking a running offset inside one block.

    'mixed' is the same question one step along: three of its operands are
    gathers this graph computes, so their launches write their bands directly,
    and the fourth is handed over and needs a copy. Carved and copied operands
    of one concat have to reach the result the same way they would alone.

    'read' and 'readMixed' are the other half of the rule. The bands are written
    by launches beside the concat rather than inside it, so anything that reads
    the result has to run after that whole step: the concat is a kernel break,
    and an op reading it lands in a later one.

    Inputs:
        a0..a5 -- six 1-D operands of differing lengths, joined as they arrive
        tail -- the operand of 'mixed' the graph hands over
        d0..d2 (64 longs each) and reps (64 longs) -- three 1-D gathers
    Outputs:
        handed: cat of the six, every band filled by a copy
        read: a reader of 'handed', which must not fuse into its kernel
        mixed: cat of three gathers and 'tail', three bands carved and one
               copied
        readMixed: a reader of 'mixed'
    """

    def forward(
        self,
        a0: Tensor,
        a1: Tensor,
        a2: Tensor,
        a3: Tensor,
        a4: Tensor,
        a5: Tensor,
        tail: Tensor,
        d0: Tensor,
        d1: Tensor,
        d2: Tensor,
        reps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        handed = torch.cat([a0, a1, a2, a3, a4, a5])
        read = handed * 2

        r0 = torch.repeat_interleave(d0, reps)
        r1 = torch.repeat_interleave(d1, reps)
        r2 = torch.repeat_interleave(d2, reps)
        mixed = torch.cat([r0, r1, r2, tail])
        readMixed = mixed + 1

        return handed, read, mixed, readMixed
