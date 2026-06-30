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


class MixedTypeMinMaxTestPreproc(nn.Module):
    """Tests min/max with mixed dtypes (int32, int64).

    Inputs: a (int32 64 elements), b (int64 64 elements)
    Outputs: minimum, maximum
    """

    def forward(
        self,
        a: Tensor,
        b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return torch.minimum(a, b), torch.maximum(a, b)
