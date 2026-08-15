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


class CloneContiguousTestPreproc(nn.Module):
    """Tests clone with contiguous memory_format.

    transpose makes x non-contiguous; clone(contiguous_format) must produce a
    real contiguous copy so the following view(-1) is valid.  If the clone is
    elided, view sees the transposed (non-contiguous) layout and the flattened
    order is wrong.

    Inputs: x (8x4 float32)
    Outputs: view(clone(transpose(x)), -1) + 1
    """

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        t = x.transpose(0, 1)
        c = t.clone(memory_format=torch.contiguous_format)
        return c.view(-1) + 1.0
