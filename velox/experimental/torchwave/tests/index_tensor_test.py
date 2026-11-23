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

from torch import nn, Tensor


class IndexTensorTest(nn.Module):
    """Tests advanced-indexing (aten.index.Tensor) lowering.

    A single 1-D integer index that selects one dimension of a rank>1 source
    lowers to the fused index_select (sel0/sel1/sel2 along dims 0/1/2). A
    separated multi-index case (an index on dim 0 and dim 2 with dim 1 kept)
    is not index_select and falls back to the eager aten op; if it were wrongly
    rewritten to index_select the result 'd' would be incorrect. A 1-D source
    indexed by a 1-D boolean mask is masked_select ('e').
    """

    def forward(
        self,
        x: Tensor,
        sel0: Tensor,
        sel1: Tensor,
        sel2: Tensor,
        sep0: Tensor,
        sep2: Tensor,
        x1d: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        a = x[sel0]  # -> index_select(x, 0, sel0)
        b = x[:, sel1]  # -> index_select(x, 1, sel1)
        c = x[:, :, sel2]  # -> index_select(x, 2, sel2)
        d = x[sep0, :, sep2]  # separated advanced indices -> eager fallback
        e = x1d[mask]  # 1-D source + 1-D boolean mask -> masked_select
        return a, b, c, d, e
