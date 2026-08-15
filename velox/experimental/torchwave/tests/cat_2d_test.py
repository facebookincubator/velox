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


class Cat2dTest(nn.Module):
    """Tests aten.cat and aten.concat of 2D tensors along dim=1.

    Inputs: a (100x50 float), b (100x30 float).
    Outputs:
        o1: torch.cat([a, b], dim=1)        -> 100x80
        o2: torch.concat([a, b], dim=1)     -> 100x80
    """

    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        o1 = torch.cat([a, b], dim=1)
        o2 = torch.concat([a, b], dim=1)
        return o1, o2
