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


class ViewTest2Preproc(nn.Module):
    """View/reshape test with nan_to_num, contiguous, view, index_select, reshape.

    Inputs: data (2D float [4, 395]), indices (1D long).
    Output: reshape(index_select(view(view(contiguous(nan_to_num(data)),
            [-1]), [-1, 395]), indices, dim=1), [-1])
    """

    def forward(self, data: Tensor, indices: Tensor) -> Tensor:
        x = torch.nan_to_num(
            data,
            nan=0.0,
            posinf=3.4028234663852886e38,
            neginf=-3.4028234663852886e38,
        )
        x = x.contiguous()
        x = x.view(-1)
        x = x.view(-1, 395)
        x = torch.index_select(x, 1, indices)
        x = x.reshape(-1)
        return x
