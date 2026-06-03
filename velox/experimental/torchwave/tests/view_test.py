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


class ViewTestPreproc(nn.Module):
    """View test: masked_select followed by truncation and reshape.

    Inputs: a, b (10000 longs each), where a[i] = i % 200, b[i] = i % 100.
    Output: masked_select result truncated to a multiple of 4, then viewed as [-1, 4].
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        selected = torch.masked_select(a + b, a % 10 < 5)
        n = selected.shape[0]
        truncated = n - n % 4
        result = selected[:truncated]
        return result.view(-1, 4)
