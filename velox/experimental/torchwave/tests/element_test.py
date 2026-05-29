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


class ElementTestPreproc(nn.Module):
    """Simple element-wise preproc for testing sigmoid packaging.

    Inputs: a, b, c (10000 longs each), d, e, f (100000 longs each)
    Outputs: o1 = a + b - c, o2 = c + 3*b - a, o3 = c - a, o4 = d + 2*e, o5 = f - e + d
    """

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
        e: Tensor,
        f: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        o1 = a + b - c
        o2 = torch.add(c, b, alpha=3) - a
        o3 = c - a
        o4 = torch.add(d, e, alpha=2)
        o5 = f - e + d
        return o1, o2, o3, o4, o5
