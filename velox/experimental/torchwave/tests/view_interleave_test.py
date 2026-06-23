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


class ViewInterleaveTest(nn.Module):
    """Interleaves fused elementwise ops with view-like breaks.

    Mirrors the ROO dense-feature preproc chain, where view-like ops (view,
    slice, select.int) sit between fused elementwise code and run host-side,
    so the wave executor emits a single node with many steps that alternate
    fused kernels and view breaks:

        elementwise -> view -> view -> slice -> elementwise -> slice
                    -> elementwise -> select.int -> elementwise -> view

    x's dim 0 is exported dynamic so the row count stays symbolic, like the
    dynamic feature batches in ROO. The [:, :K] slices produce non-contiguous
    tensors that feed the next fused kernel (and, via reshape, a clone-then-view
    break), exercising elementwise and view of a non-contiguous slice.
    """

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: [R, 40] float, R dynamic.
        y = torch.nan_to_num(x) * 2.0 + 1.0  # fused elementwise kernel #1
        y = y.reshape(-1).reshape(-1, 40)  # view -> view breaks
        y = y[:, :32]  # slice dim=1 -> [R, 32] (non-contiguous)
        y = torch.sigmoid(y) - 0.25  # fused elementwise kernel #2 on the slice
        y = y[:, :20]  # slice dim=1 -> [R, 20] (non-contiguous)
        y = torch.relu(y) + 0.5  # fused elementwise kernel #3
        row = y.select(0, 0)  # select.int row 0 -> [20] break
        row = row * row + 1.0  # fused elementwise kernel #4
        return y.reshape(-1), row  # view break + select-derived output
