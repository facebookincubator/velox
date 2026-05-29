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


class SumTestPreproc(nn.Module):
    """Sum reduction tests for different sizes, dtypes, and casting.

    Inputs: float_1k, float_10k, float_100k (float32),
            int_1k, int_10k, int_100k (int64).

    Returns the zero-dim sum tensor for each size, plus cast sums
    (int->float32, float->float64) on the 1K tensors.
    """

    def forward(
        self,
        float_1k: Tensor,
        float_10k: Tensor,
        float_100k: Tensor,
        int_1k: Tensor,
        int_10k: Tensor,
        int_100k: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        # Float sums at different sizes.
        fs1k = torch.sum(float_1k)
        fs10k = torch.sum(float_10k)
        fs100k = torch.sum(float_100k)

        # Integer sums — PyTorch promotes to int64 when dtype=None.
        is1k = torch.sum(int_1k)
        is10k = torch.sum(int_10k)
        is100k = torch.sum(int_100k)

        # Cast sums on small tensors.
        cast_int_to_float = torch.sum(int_1k, dtype=torch.float32)
        cast_float_to_double = torch.sum(float_1k, dtype=torch.float64)

        return (
            fs1k,
            fs10k,
            fs100k,
            is1k,
            is10k,
            is100k,
            cast_int_to_float,
            cast_float_to_double,
        )
