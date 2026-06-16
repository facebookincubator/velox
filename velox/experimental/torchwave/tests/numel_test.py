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


class NumelTest(nn.Module):
    """Tests aten.numel: the element count of a dynamically-shaped tensor.

    x is exported with a dynamic dim 0, so numel(x) stays symbolic rather than
    folding to a constant. Exercises:
        s = a + scalar_tensor(numel(x))   (numel -> 0-d tensor -> add broadcast)
        n = numel(x)                      (numel returned directly to host)
    """

    def forward(self, a: Tensor, x: Tensor) -> tuple[Tensor, int]:
        n = x.numel()
        t = torch.ops.aten.scalar_tensor.default(n, dtype=torch.int64)
        s = a + t
        return s, n
