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


class RepeatInterleaveTestPreproc(nn.Module):
    """repeat_interleave test covering multiple type combinations in one graph.

    Inputs:
        data_long (1000 longs), repeats_long (1000 longs) - int64 data, int64 repeats
        data_long2 (1000 longs), repeats_int (1000 ints) - int64 data, int32 repeats
        data_float (500 floats), repeats_long2 (500 longs) - float data, int64 repeats
        data_small (10 longs), repeats_small (10 longs) - small input, varied repeats

    Outputs: one repeat_interleave result per pair
    """

    def forward(
        self,
        data_long: Tensor,
        repeats_long: Tensor,
        data_long2: Tensor,
        repeats_int: Tensor,
        data_float: Tensor,
        repeats_long2: Tensor,
        data_small: Tensor,
        repeats_small: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        o1 = torch.repeat_interleave(data_long, repeats_long)
        o2 = torch.repeat_interleave(data_long2, repeats_int)
        o3 = torch.repeat_interleave(data_float, repeats_long2)
        o4 = torch.repeat_interleave(data_small, repeats_small)
        return o1, o2, o3, o4
