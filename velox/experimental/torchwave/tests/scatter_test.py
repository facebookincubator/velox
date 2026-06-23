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


class ScatterTestPreproc(nn.Module):
    """Exercises slice_scatter on 2-D tensors along dim 0 and dim 1, with the
    slice 'start' supplied at runtime (a symint produced by .item()) so a test
    can inject an out-of-range start into the frame and trigger the device-side
    bounds check in __slice_scatter.

    slice_scatter is functional: it returns a copy of 'base' with the strided
    sub-range along 'dim' overwritten by 'src'; 'base' itself is unchanged. The
    torchwave lowering rewrites it to a clone of 'base' plus an in-place
    tw.slice_scatter_ that scatters each 'src' element to its strided position.

    The slice 'end' is start + L * step (L = src length along 'dim', step a
    constant), so the slice length stays constant (== src length) and only
    'start' is data dependent. 'src' values are offset far from 'base' so a
    correct scatter is unambiguous.

    out0: dim 0, base0[s0 : s0 + 4*2 : 2, :] = src0 (4 rows, step 2).
    out1: dim 1, base1[:, s1 : s1 + 4*3 : 3] = src1 (4 cols, step 3).
    """

    def forward(
        self,
        base0: Tensor,
        src0: Tensor,
        start0: Tensor,
        base1: Tensor,
        src1: Tensor,
        start1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        step0 = 2
        len0 = src0.shape[0]
        # .item() yields a runtime symint that a test can corrupt in the frame.
        # Call the aten overload directly (as select_test does): its args are
        # untyped, so the symint flows through without the int() coercion that
        # the typed torch.slice_scatter stub would require (int() on an unbacked
        # symint forces a data-dependent guard and fails export).
        s0 = start0.item()
        out0 = torch.ops.aten.slice_scatter.default(
            base0, src0, dim=0, start=s0, end=s0 + len0 * step0, step=step0
        )

        step1 = 3
        len1 = src1.shape[1]
        s1 = start1.item()
        out1 = torch.ops.aten.slice_scatter.default(
            base1, src1, dim=1, start=s1, end=s1 + len1 * step1, step=step1
        )
        return out0, out1
