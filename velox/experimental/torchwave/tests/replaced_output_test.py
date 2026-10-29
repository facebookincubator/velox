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


class ReplacedOutputTestPreproc(nn.Module):
    """Returns the output of an op the optimizer replaces with a new value.

    x.new_ones(y.numel()) with a 1-D y is rewritten to ones_like(y), whose
    output is a new value. The graph output has to follow the rewrite, as the
    offsets outputs of the ads grouped ops do.

    Inputs: x (float32), y (int64, dynamic length)
    Output: (x.new_ones(y.numel()), y + 1)
    """

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return x.new_ones(y.numel()), y + 1
