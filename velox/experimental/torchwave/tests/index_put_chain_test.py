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

from torch import nn, Tensor


class IndexPutChainTest(nn.Module):
    """Linear chain of functional index_put ops over a new_ones tensor.

        a = base.new_ones(...)
        b = a.index_put([idx2], vals2)
        c = b.index_put([idx3], vals3)
        d = c.index_put([idx4], vals4)
        e = d * 2

    Each value in the chain is consumed by exactly one following op and never
    read again, so a, b, c and d are all reusable last uses. Exercises the
    (alias-aware) last-use flagging in ParallelExpr::computeLastUse.
    """

    def forward(
        self,
        base: Tensor,
        idx2: Tensor,
        vals2: Tensor,
        idx3: Tensor,
        vals3: Tensor,
        idx4: Tensor,
        vals4: Tensor,
    ) -> Tensor:
        a = base.new_ones(base.shape)
        b = a.index_put([idx2], vals2)
        c = b.index_put([idx3], vals3)
        d = c.index_put([idx4], vals4)
        e = d * 2
        return e
