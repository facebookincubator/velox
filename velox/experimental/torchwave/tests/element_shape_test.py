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


class ElementShapeTest(nn.Module):
    """Elementwise ops on tensors with broadcast-compatible shapes.

    Tests broadcast correctness across shapes:
    [10, 9, 8], [1, 9, 8], [10, 1, 8], [10, 9, 1], [1, 1, 1], []
    """

    def forward(
        self,
        full: Tensor,
        bc_dim0: Tensor,
        bc_dim1: Tensor,
        bc_dim2: Tensor,
        bc_all: Tensor,
        scalar: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        o1 = full + bc_dim0
        o2 = full * bc_dim1
        o3 = full - bc_dim2
        o4 = full + bc_all
        o5 = full + scalar
        return o1, o2, o3, o4, o5
