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


class ArangeTestPreproc(nn.Module):
    """Tests arange with dynamic and static sym_size.

    Input: y (1000 int64, values 0..999).
    Outputs:
        o1: arange(sym_size(masked_select(y, y % 10 < 8), dim=0))
        o2: arange(sym_size(y, dim=0))
        o3: a dynamic-size arange consumed by an elementwise expression. The
            arange length is a runtime scalar (the masked_select size), so a
            kernel boundary lets the host allocate the result, and the arange
            result dims drive the size of the fused elementwise expr.
        o4: arange(0, 5000, 5), the strided row starts of a [1000, 5] table.
        o5: arange(0, 1000, 1) plus y, step 1 given explicitly.
        o6: arange(2, 100, 7), a step that does not divide the span.
    """

    def forward(
        self, y: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = torch.masked_select(y, y % 10 < 8)
        o1 = torch.arange(x.size(0), dtype=torch.long)
        o2 = torch.arange(y.size(0), dtype=torch.long)
        o3 = torch.arange(x.size(0), dtype=torch.long) * 2 + x
        o4 = torch.arange(0, 5000, 5, dtype=torch.int32, device=y.device)
        o5 = torch.arange(0, 1000, 1, dtype=torch.int32, device=y.device) + y
        o6 = torch.arange(2, 100, 7, dtype=torch.long, device=y.device) * 3
        return o1, o2, o3, o4, o5, o6
