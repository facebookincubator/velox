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


class DedupTest(nn.Module):
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        x3: Tensor,
        x4: Tensor,
    ) -> tuple[Tensor, Tensor]:
        a = torch.cat(
            [torch.zeros(110, 10), x1, torch.ones(1, 10), x2, torch.zeros(13, 10)],
            dim=0,
        )
        b = torch.cat(
            [torch.zeros(110, 6), x3, torch.ones(1, 6), x4, torch.zeros(11, 6)],
            dim=0,
        )
        return a, b
