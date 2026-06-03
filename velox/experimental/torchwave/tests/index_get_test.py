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


class IndexGetTest(nn.Module):
    """Tests 1D and 2D index gather fused with elementwise.

    Exercises:
    1. Simple 1D index: source[indices]
    2. Chained 1D index: source[indices][indices2]
    3. Index with arithmetic on indices: source[clamp(indices * 3, 0, n-1)]
    4. Combined: source_a[idx_a] + source_b[idx_b]
    5. Index + scalar arithmetic: source[indices] + 2
    6. 2D index: matrix[row_idx, col_idx]
    7. 2D index + arithmetic: matrix[row_idx, col_idx] * 2 + 1
    """

    def forward(
        self,
        source_a: Tensor,
        source_b: Tensor,
        idx_a: Tensor,
        idx_b: Tensor,
        idx_c: Tensor,
        matrix: Tensor,
        row_idx: Tensor,
        col_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        c1 = source_a[idx_a]
        c2 = source_a[idx_a][idx_b]
        c3 = source_a[torch.clamp(idx_a * 3 + 2, 0, source_a.shape[0] - 1)]
        c4 = source_a[idx_a] + source_b[idx_b]
        c5 = source_a[idx_c] + 2
        c6 = matrix[row_idx, col_idx]
        c7 = matrix[row_idx, col_idx] * 2 + 1
        return c1, c2, c3, c4, c5, c6, c7
