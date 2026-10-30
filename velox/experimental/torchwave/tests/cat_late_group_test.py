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


class CatLateGroupTest(nn.Module):
    """A wide concat laid out in an earlier node than the one it runs in.

    The ROO preproc's cat of an unbind: 'x' is viewed as 11 rows and unbound,
    so the 11 operands are prim.ListUnpack outputs, which the first node writes
    past its kernel grids. The sort is an eager standalone, so everything after
    it is a later node, and the concat lands there because it is added to
    'late'. Placement lays the concat's allocation group out right after the
    unpack, a step past the first node's last launch. That group has to be
    carved even though nothing launches at its step: its bands are where the
    later node's copies of the 11 operands go.

    Inputs:
        x -- 11 * 256 longs, the rows the concat joins
        y -- 11 * 256 longs, sorted to force the node break
    Outputs:
        early: x * 3, so the first node has a launch of its own
        joined: cat(unbind(x.view(11, 256))) + sort(y) * 2
    """

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        early = x * 3
        parts = torch.unbind(x.view(11, 256), 0)
        sorted_y, _ = torch.sort(y)
        late = sorted_y * 2
        joined = torch.cat(parts) + late
        return early, joined
