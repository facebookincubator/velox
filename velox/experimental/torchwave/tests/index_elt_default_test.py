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


class IndexEltDefaultTestPreproc(nn.Module):
    """Advanced index over cat(var, factory), lowering to index_elt_one_default.

    matched:   cat([var, zeros])[idx] -- exactly two cat operands, the second a
               constant-fill factory. The rewrite fires: it becomes a fused
               index of 'var' with default 0, so out-of-range indices (into the
               zeros region) read 0 without materializing the cat.
    unmatched: cat([var, var2, zeros])[idx] -- three operands, so the pattern
               does not match; it stays a fused index over the materialized cat.
    """

    def forward(self, var: Tensor, var2: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
        pad = torch.zeros(4, dtype=var.dtype)
        matched = torch.cat([var, pad])[idx]
        unmatched = torch.cat([var, var2, pad])[idx]
        return matched, unmatched
