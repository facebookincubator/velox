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

from torch import nn, Tensor


class NoneOutputTestPreproc(nn.Module):
    """Returns tensors interleaved with None outputs, as the IG inference
    preprocs do.

    The executor's user outputs include the None slots while the graph's output
    node lists only the tensors, so pairing the two by position put a tensor
    into a None slot. Only an output the frame keeps after moving the outputs
    out can land there, such as a graph input returned as an output (x).

    Input: x (int32)
    Output: (x + 1, None, x, None, None, x * 2)
    """

    def forward(self, x: Tensor) -> tuple[Tensor, None, Tensor, None, None, Tensor]:
        return x + 1, None, x, None, None, x * 2
