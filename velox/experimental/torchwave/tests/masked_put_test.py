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


class MaskedPutTest(nn.Module):
    """Tests conditional assignment on float and long tensors.

    Exercises:
    1. where(flags, values, source) on float
    2. where(flags, values, source) on long
    3. Chained: where(flags2, input, where(flags1, values, source) + input)
    4. Broadcast: where with scalar mask (True) and scalar value
    5. 2D index_put_ with int indices on a [90, 130] tensor
    """

    def forward(
        self,
        source_f: Tensor,
        source_l: Tensor,
        values_f: Tensor,
        values_l: Tensor,
        flags1: Tensor,
        flags2: Tensor,
        input_f: Tensor,
        scalar_val: Tensor,
        scalar_mask: Tensor,
        dest2d: Tensor,
        idx2d_0: Tensor,
        idx2d_1: Tensor,
        vals2d: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        c1 = torch.where(flags1, values_f, source_f)
        c2 = torch.where(flags2, values_l, source_l)
        c3 = torch.where(flags1, values_f, source_f) + input_f
        c4 = torch.where(flags2, input_f, c3)
        c5 = torch.where(scalar_mask, scalar_val, source_f)

        c6 = dest2d.clone()
        c6.index_put_([idx2d_0, idx2d_1], vals2d)

        return c1, c2, c4, c5, c6
