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


class RowsNdTestPreproc(nn.Module):
    """cumsum, sum and min along the last dim of inputs of rank 2 to 4, which run
    as tw.cumsum_rows, tw.sum_rows and tw.min_rows with a row per index into
    the leading dims. The ads preproc graph has a cumsum(dim=2) of a rank-3
    int64 tensor, a sum(dim=[2]) of a rank-3 bool tensor and a min(dim=1) of a
    rank-2 int64 tensor. A transposed input checks that rows are found through
    the strides of every leading dim; small integer values make min's ties
    common, so its indices must be those of the first minimum.
    """

    def forward(
        self, x3: Tensor, b3: Tensor, x4: Tensor, f3: Tensor, m2: Tensor
    ) -> tuple[Tensor, ...]:
        x3t = x3.transpose(0, 1)
        return (
            torch.cumsum(x3, dim=2),
            torch.sum(b3, dim=[2]),
            torch.cumsum(x4, dim=-1),
            torch.sum(x4, dim=[3]),
            torch.sum(f3, dim=[-1]),
            torch.cumsum(x3t, dim=2),
            torch.sum(x3t, dim=[2]),
            *torch.min(m2, dim=1),
            *torch.min(x3t, dim=-1),
            *torch.min(f3, dim=2),
        )


def make_inputs() -> tuple[Tensor, ...]:
    g = torch.Generator().manual_seed(7)
    x3 = torch.randint(-5, 50, (4, 5, 700), generator=g)
    b3 = torch.randint(0, 2, (4, 5, 300), generator=g).to(torch.bool)
    x4 = torch.randint(-5, 50, (2, 3, 4, 260), generator=g)
    # Small integers, so float sums are exact in any order.
    f3 = torch.randint(-8, 8, (3, 7, 129), generator=g).to(torch.float32)
    m2 = torch.randint(0, 4, (64, 1100), generator=g)
    return (x3, b3, x4, f3, m2)
