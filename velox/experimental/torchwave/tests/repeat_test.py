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

from __future__ import annotations

from torch import nn, Tensor


class RepeatTestPreproc(nn.Module):
    """Tests fused elementwise aten.repeat (tiling gather by modulo).

    Exercises different ranks and repeat patterns as separate outputs:
    1. A 1-D tensor tiled by [2] (same rank) and by [2, 3] (rank expansion, a
       leading 1-dim is prepended before self).
    2. A 2-D tensor tiled by [2, 2], [1, 3] (only the inner dim grows), and
       [3, 1] (only the outer dim grows).
    3. A 3-D tensor tiled by [1, 2, 1] (only the middle dim grows).
    4. Constant-1 identities that len-match self's rank ([1] on a 1-D input and
       [1, 1] on a 2-D input); these exercise the maybeReplace identity path,
       which rewrites the repeat to its input.
    """

    def forward(
        self,
        vec: Tensor,
        mat: Tensor,
        cube: Tensor,
        id_vec: Tensor,
        id_mat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        o1 = vec.repeat(2)
        o2 = vec.repeat(2, 3)
        o3 = mat.repeat(2, 2)
        o4 = mat.repeat(1, 3)
        o5 = mat.repeat(3, 1)
        o6 = cube.repeat(1, 2, 1)
        o7 = id_vec.repeat(1)
        o8 = id_mat.repeat(1, 1)
        return o1, o2, o3, o4, o5, o6, o7, o8
