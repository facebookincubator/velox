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


class IndexListpackReuseTest(nn.Module):
    """Reproduces the cross-ProjectNode prim.ListPack-input bug.

    `g = base[gather_idx]` is an integer-index gather that lowers to the
    elementwise `tw.index_elt_one`. Because `g` is used as an index in TWO
    places (another gather and an index_put), it is materialized in its own,
    earlier ProjectNode instead of being inlined.

    The index_put `d[g] = ...` lowers to the elementwise `tw.index_put_elt_one`
    and takes `g` through a `prim.ListPack`. The fused elementwise subgraph
    therefore imports `g` from a previous kernel via a ListPack element. The
    partitioner must register that ListPack-nested cross-node value as a
    subgraph input; before the fix `extractSubgraphInputs` stops at the ListPack
    and never collects `g`, so codegen aborts with
    "Input value not found in inputs vector".
    """

    def forward(
        self,
        src: Tensor,
        base: Tensor,
        gather_idx: Tensor,
        dest: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Integer gather -> tw.index_elt_one; g is Long and used as an index in
        # BOTH ops below, so it has two consumers and is materialized in an
        # earlier ProjectNode instead of being inlined into each use.
        g = base[gather_idx]
        # Use 1: a gather with g as the index (g via prim.ListPack).
        gathered = src[g] + 1.0
        # Use 2: an in-place index_put with g as the index (g via prim.ListPack),
        # whose result feeds a downstream elementwise (d + 3). That downstream
        # fuse pulls tw.index_put_elt_one into a fused elementwise node which must
        # import its cross-node inputs (the clone self and the ListPack index g)
        # from the earlier ProjectNode -- the collection gap under test.
        d = dest.clone()
        d[g] = values * 2.0 + 1.0
        return gathered, d + 3.0
