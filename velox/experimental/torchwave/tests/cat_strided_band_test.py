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


class CatStridedBandTest(nn.Module):
    """Concats whose copied operands land in a pitched band of the result.

    A clone is a copy: it reads its source at the source's layout and writes
    one element per element. Joined off the outermost axis, the region it
    occupies is a band of the result carrying the result's pitch, not a run, so
    the copy has to map its write through the destination's strides. A copy
    that indexes the destination linearly walks out of its band and into the
    next operand's, which is why o3 puts a copy between two computed operands:
    there the corruption shows up in a neighbour and not only as a short tail
    of its own.

    Every source is transposed, because a clone survives only as a real layout
    conversion -- one whose input is already contiguous is elided as an
    identity. That also makes each copy strided on both sides at once.
    torch.clone(..., memory_format=...) rather than .contiguous(): the latter
    exports to aten.contiguous.default, which is a registration of its own.

    Every clone is read a second time by 'marks', which is what materializes it
    in a launch of its own instead of folding it into the concat's kernel. Only
    an operand some launch writes is one the concat has a write to redirect.

    Inputs: a..i (4x6 floats), p, q, r (2x3x4 floats).
    Outputs:
        o1: three copies joined on dim 1 -> 6x12, every band pitched.
        o2: three copies joined on the innermost dim of a rank-3 result ->
            2x4x9, where the band has two strides to get right, not one.
        o3: a copy between two computed operands on dim 1 -> 6x12.
        o4: three copies on dim 0 -> 18x4. The band is a run here, so this is
            the contiguous path the copy must keep taking.
        marks: the second read of every copy.
    """

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
        e: Tensor,
        f: Tensor,
        g: Tensor,
        h: Tensor,
        i: Tensor,
        p: Tensor,
        q: Tensor,
        r: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        fmt = torch.contiguous_format
        ca = torch.clone(a.t(), memory_format=fmt)
        cb = torch.clone(b.t(), memory_format=fmt)
        cc = torch.clone(c.t(), memory_format=fmt)
        cd = torch.clone(d.t(), memory_format=fmt)
        ce = torch.clone(e.t(), memory_format=fmt)
        cf = torch.clone(f.t(), memory_format=fmt)
        cg = torch.clone(g.t(), memory_format=fmt)
        ch = torch.clone(h.t(), memory_format=fmt)
        ci = torch.clone(i.t(), memory_format=fmt)
        cp = torch.clone(p.transpose(1, 2), memory_format=fmt)
        cq = torch.clone(q.transpose(1, 2), memory_format=fmt)
        cr = torch.clone(r.transpose(1, 2), memory_format=fmt)

        o1 = torch.cat([ca, cb, cc], dim=1)
        o2 = torch.cat([cp, cq, cr], dim=2)
        o3 = torch.cat([cd * 2.0, ce, cf - 1.0], dim=1)
        o4 = torch.cat([cg, ch, ci], dim=0)

        marks2 = ca + cb + cc + cd + ce + cf + cg + ch + ci
        marks3 = cp + cq + cr

        return o1, o2, o3, o4, marks2, marks3
