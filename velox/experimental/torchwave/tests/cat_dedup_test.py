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


class CatDedupTest(nn.Module):
    """Two wide concats of identical shape over different operands.

    The two have the same structure -- a gather, an elementwise, and two graph
    inputs joined in the same order -- so they deduplicate to a single project
    operation that runs twice. Everything the concat needs is then formal, and
    every instance has to be given its own actual: the result, the operands the
    allocation group carves, and the values the copies write, which belong to
    no graph node the caller can see.

    'shared' is joined by BOTH concats, and twice within each. That is four
    regions filled from one buffer. Each needs a copy and a destination of its
    own; one destination reused across the four would leave three regions
    holding whatever the result was allocated over.

    Inputs:
        d0, d1 (64 longs) and reps (64 longs) -- the gathered operands
        shared (16 longs) -- joined twice by each concat, and never written by
            any kernel, so it can only reach its regions by being copied
        tail0, tail1 (24 longs) -- one graph input per concat, so the two
            concats are over different operands rather than the same ones
    Outputs:
        first, second: the two concats
        marks: a second consumer of each gather, so neither folds into its
            concat's kernel and both cross a boundary the way the ROO gathers do
    """

    def forward(
        self,
        d0: Tensor,
        d1: Tensor,
        reps: Tensor,
        shared: Tensor,
        tail0: Tensor,
        tail1: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r0 = torch.repeat_interleave(d0, reps)
        r1 = torch.repeat_interleave(d1, reps)

        first = torch.cat([r0, shared, r0 * 2, shared, tail0])
        second = torch.cat([r1, shared, r1 * 2, shared, tail1])

        marks = r0 + r1
        return first, second, marks
