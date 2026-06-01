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


class IndexTest(nn.Module):
    """Tests index_put_ with clone on 1D, 2D, 3D, and broadcast cases.

    2D uses int32 for dim0, long for dim1.
    3D uses int32 for dim0 and dim2, long for dim1.
    Broadcast case: 3D dest with a scalar value, a scalar index on dim0,
    a strided (stride=2) index on dim1, and a normal index on dim2.
    """

    def forward(
        self,
        dest1d: Tensor,
        idx1d_0: Tensor,
        vals1d: Tensor,
        dest2d: Tensor,
        idx2d_0: Tensor,
        idx2d_1: Tensor,
        vals2d: Tensor,
        dest3d: Tensor,
        idx3d_0: Tensor,
        idx3d_1: Tensor,
        idx3d_2: Tensor,
        vals3d: Tensor,
        dest_bc: Tensor,
        idx_bc_0: Tensor,
        idx_bc_1: Tensor,
        idx_bc_2: Tensor,
        vals_bc: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        c1 = dest1d.clone()
        c1.index_put_([idx1d_0], vals1d)

        c2 = dest2d.clone()
        c2.index_put_([idx2d_0, idx2d_1], vals2d)

        c3 = dest3d.clone()
        c3.index_put_([idx3d_0, idx3d_1, idx3d_2], vals3d)

        c4 = dest_bc.clone()
        c4.index_put_([idx_bc_0, idx_bc_1, idx_bc_2], vals_bc)

        return c1, c2, c3, c4
