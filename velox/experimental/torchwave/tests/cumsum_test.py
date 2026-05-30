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


class CumsumTestPreproc(nn.Module):
    """Cumulative sum tests for different sizes and dtypes.

    Inputs: float_1k, float_10k, float_100k (float32),
            int_1k, int_10k, int_100k (int64).

    Returns cumsum(dim=0) for each input.
    """

    def forward(
        self,
        float_1k: Tensor,
        float_10k: Tensor,
        float_100k: Tensor,
        int_1k: Tensor,
        int_10k: Tensor,
        int_100k: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        return (
            torch.cumsum(float_1k, dim=0),
            torch.cumsum(float_10k, dim=0),
            torch.cumsum(float_100k, dim=0),
            torch.cumsum(int_1k, dim=0),
            torch.cumsum(int_10k, dim=0),
            torch.cumsum(int_100k, dim=0),
        )
