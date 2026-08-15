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
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from torch import nn, Tensor


class TypesTestPreproc(nn.Module):
    """Exercises arithmetic type promotion where PyTorch differs from C++.

    Each output computes a + b * c with a type combination that triggers
    PyTorch promotion rules that differ from C++ implicit conversions.

    Case 1 (Long * Half):
        PyTorch promotes both to Float; C++ converts Long to Half.
        Values > 2048 lose precision in Half but not Float.

    Case 2 (Long * BFloat16):
        PyTorch promotes both to Float; C++ converts Long to BFloat16.
        Values > 256 lose precision in BFloat16 but not Float.

    Case 3 (Int * Half):
        PyTorch promotes both to Float; C++ converts Int to Half.
        Same precision boundary as Case 1.

    Case 4 (Long * Float):
        PyTorch and C++ both produce Float.  Included as a baseline that
        should always match.
    """

    def forward(
        self,
        a1: Tensor,
        b1: Tensor,
        c1: Tensor,
        a2: Tensor,
        b2: Tensor,
        c2: Tensor,
        a3: Tensor,
        b3: Tensor,
        c3: Tensor,
        a4: Tensor,
        b4: Tensor,
        c4: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        o1 = a1 + b1 * c1  # long + long * half -> float
        o2 = a2 + b2 * c2  # long + long * bfloat16 -> float
        o3 = a3 + b3 * c3  # int  + int  * half -> float
        o4 = a4 + b4 * c4  # long + long * float -> float
        return o1, o2, o3, o4
