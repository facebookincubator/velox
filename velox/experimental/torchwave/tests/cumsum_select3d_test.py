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


class CumsumSelect3dTestPreproc(nn.Module):
    """cumsum over a doubly-strided view, matching the ads-preproc range
    pattern cumsum(select(select(x, dim=1, index=0), dim=1, index=1), dim=0).

    Two chained selects on a 3D tensor produce a 1D view with a large stride
    (rows of x) and a storage offset.  The scan must honor that stride / not
    mistake the view for contiguous; otherwise it reads x's row-major storage
    and sums the wrong elements.

    Input: x (int64, 3D, multi-block row count)
    Output: cumsum(x[:, 0, 1], dim=0)
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.cumsum(x[:, 0, 1], dim=0)
