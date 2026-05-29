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


class IsinTest(nn.Module):
    """Tests torch.isin with int32 and int64 tensors of various sizes.

    Each call returns a 1D bool tensor indicating which elements of the
    first argument are present in the second argument (the set).
    """

    def forward(
        self,
        elements_i32_1k: Tensor,
        set_i32_1k: Tensor,
        elements_i64_1k: Tensor,
        set_i64_1k: Tensor,
        elements_i32_10k: Tensor,
        set_i32_10k: Tensor,
        elements_i64_100k: Tensor,
        set_i64_100k: Tensor,
        elements_i32_nz: Tensor,
        set_i32_with_zero: Tensor,
        elements_i64_nz: Tensor,
        set_i64_with_zero: Tensor,
        zeros_i64: Tensor,
        single_zero_i64: Tensor,
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
    ]:
        r1 = torch.isin(elements_i32_1k, set_i32_1k)
        r2 = torch.isin(elements_i64_1k, set_i64_1k)
        r3 = torch.isin(elements_i32_10k, set_i32_10k)
        r4 = torch.isin(elements_i64_100k, set_i64_100k)
        r5 = torch.isin(elements_i32_nz, set_i32_with_zero)
        r6 = torch.isin(elements_i64_nz, set_i64_with_zero)
        r7 = torch.isin(elements_i32_1k, set_i32_1k, assume_unique=True, invert=True)
        r8 = torch.isin(elements_i64_1k, set_i64_1k, invert=True)
        r9 = torch.isin(zeros_i64, single_zero_i64)
        return r1, r2, r3, r4, r5, r6, r7, r8, r9
