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


class MaskedPutTest(nn.Module):
    """Tests conditional assignment on float and long tensors.

    Exercises:
    1. where(flags, values, source) on float
    2. where(flags, values, source) on long
    3. Chained: where(flags2, input, where(flags1, values, source) + input)
    4. Broadcast: where with scalar mask (True) and scalar value
    5. 2D index_put_ with int indices on a [90, 130] tensor
    6. In-place bool-mask index_put_ on scalar_val (an input that is also read
       by c5 before the mutation and returned as c7 after it), plus a functional
       index_put with the negated mask.
    7. Functional (out-of-place) index_put with numeric indices and, separately,
       with a bool mask: each returns the unmodified source alongside the
       result, so the clone-based index_put rewrite is verified to leave the
       original input untouched.
    """

    def forward(
        self,
        source_f: Tensor,
        source_l: Tensor,
        values_f: Tensor,
        values_l: Tensor,
        flags1: Tensor,
        flags2: Tensor,
        input_f: Tensor,
        scalar_val: Tensor,
        scalar_mask: Tensor,
        scalar_flags: Tensor,
        dest2d: Tensor,
        idx2d_0: Tensor,
        idx2d_1: Tensor,
        vals2d: Tensor,
        ip_num_src: Tensor,
        ip_num_idx: Tensor,
        ip_num_vals: Tensor,
        ip_bool_src: Tensor,
        ip_bool_mask: Tensor,
        ip_bool_vals: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        c1 = torch.where(flags1, values_f, source_f)
        c2 = torch.where(flags2, values_l, source_l)
        c3 = torch.where(flags1, values_f, source_f) + input_f
        c4 = torch.where(flags2, input_f, c3)
        c5 = torch.where(scalar_mask, scalar_val, source_f)

        c6 = dest2d.clone()
        c6.index_put_([idx2d_0, idx2d_1], vals2d)

        c7 = scalar_val
        c8 = scalar_val.index_put_([scalar_flags], scalar_val + 1)
        c9 = scalar_val.index_put([torch.bitwise_not(scalar_flags)], scalar_val + 10)

        # Functional index_put: assign elements and return the original source
        # plus the result, with numeric indices and with a bool mask.
        c10 = ip_num_src.index_put([ip_num_idx], ip_num_vals)
        c11 = ip_bool_src.index_put([ip_bool_mask], ip_bool_vals)

        return c1, c2, c4, c5, c6, c7, c8, c9, ip_num_src, c10, ip_bool_src, c11
