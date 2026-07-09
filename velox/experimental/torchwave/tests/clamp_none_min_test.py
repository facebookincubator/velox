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


class ClampNoneMinTestPreproc(nn.Module):
    """Clamp with min=None (an absent optional, carried as a None-typed input
    value) and a constant max.

    Reproduces the ads-preproc bug where a None min was treated as present and
    filled with 0, so the kernel computed min(max(x, 0), 5) and wrongly clamped
    negative inputs up to 0 instead of passing them through (correct: min(x, 5)).

    Input: x (int32, contains negatives)
    Output: clamp(x, min=None, max=5)
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=None, max=5)
