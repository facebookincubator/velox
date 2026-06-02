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


class CatTest2Preproc(nn.Module):
    """Cat with view (select) and masked_select inputs.

    Inputs: x (2x1000 int64), y (300 int64).
    """

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        s0 = torch.ops.aten.select.int(x, dim=0, index=0)
        s1 = torch.ops.aten.select.int(x, dim=0, index=1)
        o1 = torch.cat([y * 3, s0, s1, y + 10], dim=0)

        ms = torch.masked_select(y, y % 10 < 8)
        o2 = torch.cat([ms, s0, s1, y + 10], dim=0)

        return o1, o2
