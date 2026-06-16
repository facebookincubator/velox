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


def _group(t: Tensor) -> Tensor:
    # narrow -> fused elementwise / contiguous -> flatten + reshape (view
    # chain) -> concat back along dim=1.
    a = t.narrow(1, 0, 50)
    b = t.narrow(1, 50, 30)
    a = torch.clamp(a, -1.0, 1.0)
    b = b.contiguous()
    a = a.reshape(-1).reshape(100, 50)
    b = b.reshape(-1).reshape(100, 30)
    return torch.concat([a, b], dim=1)


class Cat2dViewTest(nn.Module):
    """Chained same-shape concats: o2 depends on o1 so the two concats land in
    different waves (ProjectNodes). The second reuses the first's standalone
    concat ProjectOperation across ProjectNode boundaries."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        o1 = _group(x)
        o2 = _group(o1)
        return o1, o2
