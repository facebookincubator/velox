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


class BincountTestPreproc(nn.Module):
    """bincount reduction tests.

    Inputs are 1-D non-negative int64 index tensors. Exercises a small input
    (single-block final), a large input (multi-block / cg final, with heavy
    atomic-add contention on a modest bin count), and minlength padding beyond
    the observed max. The output length is data-dependent: max(input) + 1, at
    least minlength.
    """

    def forward(
        self,
        small: Tensor,
        large: Tensor,
    ) -> tuple[Tensor, ...]:
        b_small = torch.bincount(small)
        b_large = torch.bincount(large)
        b_minlen = torch.bincount(small, minlength=16)
        return (
            b_small,
            b_large,
            b_minlen,
        )
