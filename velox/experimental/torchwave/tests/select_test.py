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


class SelectTest(nn.Module):
    """Tests torch.select with constant and computed indices on 1-3D tensors.

    Inputs:
      t1d: 1D tensor of longs (used both as data and as index source)
      t2d: 2D tensor of longs
      t3d: 3D tensor of longs

    The indices tensor provides on-device index values: selecting an element
    from it with a constant index and calling .item() yields a SymInt that
    the compiler must handle as a runtime scalar.
    """

    def forward(
        self,
        t1d: Tensor,
        t2d: Tensor,
        t3d: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Constant-index selects.
        r1 = torch.ops.aten.select.int(t1d, 0, 2)
        r2 = torch.ops.aten.select.int(t2d, 0, 1)
        r3 = torch.ops.aten.select.int(t2d, 1, 3)
        r4 = torch.ops.aten.select.int(t3d, 0, 0)
        r5 = torch.ops.aten.select.int(t3d, 2, 2)

        # On-device computed index: read from indices tensor.
        idx0 = torch.ops.aten.select.int(indices, 0, 0).item()
        r6 = torch.ops.aten.select.int(t2d, 0, idx0)

        idx1 = torch.ops.aten.select.int(indices, 0, 1).item()
        r7 = torch.ops.aten.select.int(t3d, 1, idx1)

        return r1, r2, r3, r4, r5, r6, r7
