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


class ElementShapeTest3(nn.Module):
    """Three-way broadcast across operands of different rank.

    t1 is [100], t2 is [20, 1], t3 is [10, 1, 1]; t1 + t2 + t3 broadcasts to
    [10, 20, 100]. The caller initializes each tensor with arange and reshapes
    it to its final shape before calling forward.
    """

    def forward(self, t1: Tensor, t2: Tensor, t3: Tensor) -> Tensor:
        return t1 + t2 + t3
