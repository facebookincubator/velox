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

import torch
from torch import nn, Tensor


def _cumsum_of_cast(x: Tensor, off: int) -> Tensor:
    # cumsum of a select-view of a wave-produced cast.  The select-view of a
    # graph input does not exercise the bug -- the producer must itself be a
    # wave op (the cast) so it can be fused into the scan kernel.  Distinct
    # `off` per chain defeats subgraph dedup.
    base = (x + off).to(torch.int32)
    sel = base[:, 0, 1]
    return torch.cumsum(sel, 0, dtype=torch.long)


class CumsumFusedProducerTestPreproc(nn.Module):
    """Many concurrent cumsum(select(cast)) chains feeding a cat.  The scale is
    load-bearing: enough chains are needed for the fused cast producer's
    per-block writes to lose the race against the scan's cross-block reads."""

    def forward(self, x: Tensor) -> Tensor:
        outs = [_cumsum_of_cast(x, off) for off in range(64)]
        return torch.cat(outs)
