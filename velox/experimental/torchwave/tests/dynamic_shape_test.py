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
    """Tests data-dependent dynamic shapes: arange from sum, ones from length values.

    Inputs: lengths (20 int64 values between 1 and 4)
    Outputs: s (scalar sum), r (arange(s)), o (ones shaped by lengths[2] x lengths[3])
    """

    def forward(
        self,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        s = torch.sum(lengths)
        r = torch.arange(s.item())
        o = torch.ones(size=[int(lengths[2].item()), int(lengths[3].item())])
        return s, r, o
