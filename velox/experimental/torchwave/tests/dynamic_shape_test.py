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


class DynamicShapeTestPreproc(nn.Module):
    """Tests _local_scalar_dense (Tensor.item()) feeding a dynamic size.

    r = arange(item(sum(lengths))) has a data-dependent length whose end is a
    runtime scalar produced by _local_scalar_dense; r2 = r * 2 + 1 then consumes
    that arange in a fused elementwise expression, so the arange's dynamic dims
    drive the expr size. This exercises item() producing a scalar (fused like
    aten.item), not a zero-dim tensor.

    Inputs: lengths (20 int64 values between 1 and 4)
    Outputs: s (scalar sum), r (arange(s.item())), r2 (r * 2 + 1)
    """

    def forward(
        self,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        s = torch.sum(lengths)
        r = torch.arange(s.item(), dtype=torch.long)
        r2 = r * 2 + 1
        return s, r, r2
