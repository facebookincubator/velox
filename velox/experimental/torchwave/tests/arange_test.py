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
    """

    def forward(self, y: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.masked_select(y, y % 10 < 8)
        o1 = torch.arange(x.size(0), dtype=torch.long)
        o2 = torch.arange(y.size(0), dtype=torch.long)
        return o1, o2
