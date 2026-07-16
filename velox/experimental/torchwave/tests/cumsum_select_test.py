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


class CumsumSelectTestPreproc(nn.Module):
    """cumsum(dim=0) over a non-contiguous column view, matching the
    ads-preproc pattern cumsum(select(x, dim=1, index=1), dim=0).

    The select produces a strided view (stride = ncols); the cumsum must honor
    that stride rather than reading the backing storage contiguously.  With a
    contiguous misread the running sum is computed over the wrong elements.

    Input: x (int64, 2D, multi-block row count)
    Output: cumsum(x[:, 1], dim=0)
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.cumsum(x[:, 1], dim=0)
