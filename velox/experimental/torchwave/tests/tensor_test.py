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


class TensorTest(nn.Module):
    """Tests scalar_tensor (aten.tensor of a symbolic scalar) and _to_copy.

    n = sym_size(x, 0) is symbolic (x's dim 0 is exported as dynamic), so
    t = scalar_tensor(n) is a 0-d tensor materialized from that scalar rather
    than a folded constant. The aten ops are called directly so the exported
    graph contains exactly scalar_tensor / add / _to_copy without the extra
    to.device / detach / _assert_tensor_metadata that the torch.tensor(...) and
    Tensor.to(...) sugar would introduce. Exercises:
        s    = a + t                          (add.Tensor broadcasting the 0-d)
        a_cp = _to_copy(a, dtype=int32)        (_to_copy on the input tensor)
        t_cp = _to_copy(t, dtype=float32)      (_to_copy on the 0-d tensor)
    Both _to_copy calls change dtype.
    """

    def forward(self, a: Tensor, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n = x.shape[0]
        t = torch.ops.aten.scalar_tensor.default(n, dtype=torch.int64)
        s = a + t
        a_cp = torch.ops.aten._to_copy.default(a, dtype=torch.int32)
        t_cp = torch.ops.aten._to_copy.default(t, dtype=torch.float32)
        return s, a_cp, t_cp
