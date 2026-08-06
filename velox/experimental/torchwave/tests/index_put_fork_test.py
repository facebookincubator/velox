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


class IndexPutForkTest(nn.Module):
    """A shared value feeds two independent index_put chains; both ends escape.

        c  = a + b
        c1 = c.index_put([idx1], vals1)   # chain 1
        c2 = c1.index_put([idx2], vals2)
        c3 = c.index_put([idx3], vals3)   # chain 2
        c4 = c3.index_put([idx4], vals4)
        return c2, c4

    c is read by both chains, so its buffer must NOT be reused in place by
    either chain. The single-use temporaries within each chain are exclusive
    last uses. Exercises the alias-aware reusable-last-use flagging in
    ParallelExpr::computeLastUse.
    """

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        idx1: Tensor,
        vals1: Tensor,
        idx2: Tensor,
        vals2: Tensor,
        idx3: Tensor,
        vals3: Tensor,
        idx4: Tensor,
        vals4: Tensor,
    ) -> tuple[Tensor, Tensor]:
        c = a + b
        c1 = c.index_put([idx1], vals1)
        c2 = c1.index_put([idx2], vals2)
        c3 = c.index_put([idx3], vals3)
        c4 = c3.index_put([idx4], vals4)
        return c2, c4
