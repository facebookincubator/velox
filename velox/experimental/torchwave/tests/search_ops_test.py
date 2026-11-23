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

from __future__ import annotations

import torch
from torch import nn, Tensor


class SearchOpsTest(nn.Module):
    """Tests fused elementwise bucketize / searchsorted binary search.

    Exercises:
    1. bucketize on a float query into 1-D float boundaries, right=False and
       right=True, with queries that hit boundary values (to pin the lower vs
       upper bound convention), out-of-range values, and a NaN.
    2. bucketize on an int query into 1-D int boundaries with out_int32=True.
    3. bucketize fused with a downstream elementwise add (the result feeds an
       arithmetic op), the common ads shape.
    4. searchsorted into a 1-D sorted sequence, right=False, right=True, and
       side="right" (which must equal right=True).
    5. searchsorted into a 2-D (multi-row) sorted sequence, where each query row
       searches the matching sorted row, right=False and right=True with
       out_int32=True.
    """

    def forward(
        self,
        query_f: Tensor,
        bounds_f: Tensor,
        query_i: Tensor,
        bounds_i: Tensor,
        query_1d: Tensor,
        sorted_1d: Tensor,
        query_2d: Tensor,
        sorted_2d: Tensor,
    ) -> tuple[Tensor, ...]:
        bucket_left = torch.bucketize(query_f, bounds_f)
        bucket_right = torch.bucketize(query_f, bounds_f, right=True)
        bucket_int32 = torch.bucketize(query_i, bounds_i, out_int32=True)
        bucket_fused = torch.bucketize(query_f, bounds_f) + 1

        search_left = torch.searchsorted(sorted_1d, query_1d)
        search_right = torch.searchsorted(sorted_1d, query_1d, right=True)
        search_side = torch.searchsorted(sorted_1d, query_1d, side="right")

        search_2d = torch.searchsorted(sorted_2d, query_2d)
        search_2d_right = torch.searchsorted(
            sorted_2d, query_2d, right=True, out_int32=True
        )

        return (
            bucket_left,
            bucket_right,
            bucket_int32,
            bucket_fused,
            search_left,
            search_right,
            search_side,
            search_2d,
            search_2d_right,
        )
