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


class PowScalarTestPreproc(nn.Module):
    """A scalar base raised to a tensor of exponents (aten.pow.Scalar), as the
    ads preproc graph has 2 ** x of a float tensor, and an empty new_ones that
    feeds a cumsum and a cat, which the ads graph also has. The pow fuses; the
    empty new_ones runs as a host shortcut with no kernel.
    """

    def forward(self, x: Tensor, n: Tensor, ids: Tensor) -> tuple[Tensor, ...]:
        empty = ids.new_ones([0])
        return (
            torch.pow(2, x),
            torch.pow(3, n),
            torch.cumsum(empty, dim=0),
            torch.cat([torch.cumsum(ids, dim=0), empty]),
        )


def make_inputs() -> tuple[Tensor, ...]:
    g = torch.Generator().manual_seed(11)
    x = torch.rand(3000, generator=g) * 20 - 10
    # 3 ** 12 is still exact in the float pow computes in.
    n = torch.randint(0, 13, (1500,), generator=g)
    ids = torch.randint(0, 100, (700,), generator=g)
    return (x, n, ids)
