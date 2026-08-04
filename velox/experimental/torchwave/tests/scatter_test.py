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


class ScatterTestPreproc(nn.Module):
    """Exercises slice_scatter on 2-D tensors along dim 0 and dim 1, with the
    slice 'start' supplied at runtime (a symint produced by .item()) so a test
    can inject an out-of-range start into the frame and trigger the device-side
    bounds check in __slice_scatter.

    slice_scatter is functional: it returns a copy of 'base' with the strided
    sub-range along 'dim' overwritten by 'src'; 'base' itself is unchanged. The
    torchwave lowering rewrites it to a clone of 'base' plus an in-place
    tw.slice_scatter_ that scatters each 'src' element to its strided position.

    The slice 'end' is start + L * step (L = src length along 'dim', step a
    constant), so the slice length stays constant (== src length) and only
    'start' is data dependent. 'src' values are offset far from 'base' so a
    correct scatter is unambiguous.

    out0: dim 0, base0[s0 : s0 + 4*2 : 2, :] = src0 (4 rows, step 2).
    out1: dim 1, base1[:, s1 : s1 + 4*3 : 3] = src1 (4 cols, step 3).
    """

    def forward(
        self,
        base0: Tensor,
        src0: Tensor,
        start0: Tensor,
        base1: Tensor,
        src1: Tensor,
        start1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        step0 = 2
        len0 = src0.shape[0]
        # .item() yields a runtime symint that a test can corrupt in the frame.
        # Call the aten overload directly (as select_test does): its args are
        # untyped, so the symint flows through without the int() coercion that
        # the typed torch.slice_scatter stub would require (int() on an unbacked
        # symint forces a data-dependent guard and fails export).
        s0 = start0.item()
        out0 = torch.ops.aten.slice_scatter.default(
            base0, src0, dim=0, start=s0, end=s0 + len0 * step0, step=step0
        )

        step1 = 3
        len1 = src1.shape[1]
        s1 = start1.item()
        out1 = torch.ops.aten.slice_scatter.default(
            base1, src1, dim=1, start=s1, end=s1 + len1 * step1, step=step1
        )
        return out0, out1


class SelectScatterTestPreproc(nn.Module):
    """Exercises the fused tw.select_scatter along dim 0 and dim 1, covering both
    the in-place-reuse path and the copy path.

    select_scatter is functional: it returns a copy of 'base' with the slice
    base.select(dim, index) replaced by 'src' (src has base's shape with 'dim'
    removed). torchwave lowers it to the fused elementwise tw.select_scatter,
    whose output reuses base's buffer in place when base is a dead intermediate,
    or is freshly allocated (a copy) when base is still live afterwards.

    'base' is an intermediate (in0 + in1), never an externally-owned graph input
    (those are never reused), so the reuse decision is driven purely by liveness:
      - out0 (in-place): base0 is consumed only by select_scatter -> its buffer
        is reused as the output.
      - out1 (copy): base1 is also returned, so it stays live and cannot be
        overwritten -> the output is a fresh copy.

    'index' is a compile-time constant so the rewrite to tw.select_scatter fires
    (dim must also be statically 0 or 1).
    """

    def forward(
        self,
        a0: Tensor,
        a1: Tensor,
        src0: Tensor,
        b0: Tensor,
        b1: Tensor,
        src1: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        base0 = a0 + a1  # dead intermediate -> select_scatter reuses it in place
        out0 = torch.ops.aten.select_scatter.default(base0, src0, dim=0, index=2)

        base1 = b0 + b1  # also returned below -> stays live -> copy
        out1 = torch.ops.aten.select_scatter.default(base1, src1, dim=1, index=3)
        return out0, out1, base1


class SliceScatterInPlaceTestPreproc(nn.Module):
    """Exercises the fused tw.slice_scatter with an INTERMEDIATE base, covering
    the in-place-reuse and copy paths (scatterTest only covers graph-input bases,
    whose clone is always kept).

    slice_scatter is functional: it returns a copy of 'base' with the strided
    sub-range along 'dim' overwritten by 'src'. torchwave lowers it to a clone of
    'base' plus the fused in-place tw.slice_scatter; clone-elision drops the clone
    when base is a dead intermediate (write lands in base's buffer) and keeps it
    when base stays live.

    'base' is an intermediate (in0 + in1), never an externally-owned graph input
    (those are never reused), so the reuse decision is driven purely by liveness:
      - out0 (in-place): base0 is consumed only by slice_scatter -> its buffer is
        reused as the output (the elided-clone / intra-op-materialized-self path).
      - out1 (copy): base1 is also returned, so it stays live and cannot be
        overwritten -> the output is a fresh copy (clone kept).

    'start'/'end'/'step' are compile-time constants and 'dim' is statically 0 or 1
    so the rewrite to tw.slice_scatter fires.
    """

    def forward(
        self,
        a0: Tensor,
        a1: Tensor,
        src0: Tensor,
        b0: Tensor,
        b1: Tensor,
        src1: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        base0 = a0 + a1  # dead intermediate -> clone elided -> write in place
        out0 = torch.ops.aten.slice_scatter.default(
            base0, src0, dim=0, start=2, end=10, step=2
        )

        base1 = b0 + b1  # also returned below -> stays live -> copy
        out1 = torch.ops.aten.slice_scatter.default(
            base1, src1, dim=1, start=1, end=13, step=3
        )
        return out0, out1, base1


class ScatterSrcTestPreproc(nn.Module):
    """Exercises the fused tw.scatter (aten.scatter.src) along dim 0 and dim 1.

    scatter.src is functional: out = self.clone() with out scattered from src at
    the coordinates in the index tensor (index and src share a shape); each
    destination takes the src coordinates with the 'dim' axis replaced by
    index[i]. torchwave lowers it to a clone of self plus the fused in-place
    tw.scatter. 'base' is an intermediate (a0 + a1) so the clone is elidable.

    The index is a permutation along 'dim' (each destination written once), so
    the overwrite result is deterministic and matches eager even though wave
    scatters in parallel. 'dim' is a constant so the rewrite to tw.scatter fires.

    out0: dim 0. out1: dim 1.
    """

    def forward(
        self,
        a0: Tensor,
        a1: Tensor,
        idx0: Tensor,
        src0: Tensor,
        b0: Tensor,
        b1: Tensor,
        idx1: Tensor,
        src1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        base0 = a0 + a1
        out0 = torch.ops.aten.scatter.src(base0, 0, idx0, src0)
        base1 = b0 + b1
        out1 = torch.ops.aten.scatter.src(base1, 1, idx1, src1)
        return out0, out1


class ScatterAddTestPreproc(nn.Module):
    """Exercises the fused tw.scatter_add (aten.scatter_add.default) along dim 0
    and dim 1, covering clone-elision.

    scatter_add accumulates with an atomic add, so duplicate destination indices
    sum and the result is order-independent (matches eager despite wave scattering
    in parallel). torchwave lowers it to a clone of self plus the fused in-place
    tw.scatter_add.
      - out0 (clone elided): base0 = a0 + a1 is a dead intermediate and src0 is an
        independent input, so the clone of base0 is dropped and the accumulation
        lands in base0's buffer.
      - out1 (clone kept): src is base1 itself, which shares base1's storage, so
        the clone must be kept -- accumulating in place would read partially
        updated values.
    """

    def forward(
        self,
        a0: Tensor,
        a1: Tensor,
        idx0: Tensor,
        src0: Tensor,
        b0: Tensor,
        b1: Tensor,
        idx1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        base0 = a0 + a1  # dead intermediate, independent src -> clone elided
        out0 = torch.ops.aten.scatter_add.default(base0, 0, idx0, src0)
        base1 = b0 + b1
        # src is base1 itself (shares base1's storage) -> clone kept.
        out1 = torch.ops.aten.scatter_add.default(base1, 1, idx1, base1)
        return out0, out1


class ScatterAddCgConsumerTestPreproc(nn.Module):
    """Repro for the cooperative-grid failure on the ROO preproc: a fused
    tw.scatter_add whose src/index are far larger than the tensor it accumulates
    into, feeding an elementwise expression sized by that smaller tensor.

    tw.scatter_add returns in memory, not in a register, so it is an elementwise
    BORDER: it is emitted as its own expression and the consumer reads its
    materialized [S] output. Under a cooperative grid both expressions land in
    one kernel op (multi-block ends the producer's kernel instead), and the size
    walk used to recurse through the border and take the scatter's [N] src/index
    as size leaves of the consumer -- so `limit` came out [N] instead of [S] and
    the device loop, sized by the output, ran N iterations over [S] operands.

    'index' is deliberately read twice: once as the scatter's index (which forces
    an own-dims index calculator on it) and again as the gather index of
    `limit[index]` in a later expression of the same kernel op. The second use
    allocates an alt Tensor copy, which the own-dims primary used to suppress,
    leaving the copy at its zero fill -- null storage the gather then read.

    Two structural details are what make the bug reachable, and both are easy to
    lose. The trailing cumsum puts the graph on the cooperative grid at all:
    makeProjectionOperation builds the cg variant only when some node in the
    ProjectOp has one (ROO had tw.masked_select_jagged_cg), and without it the
    multi-block grid is compiled even under isCg -- which ends the scatter's
    kernel and hides the bug. And the whole chain is single-use: a multi-use
    value (e.g. returning `limit` as well) is a layer boundary, which splits the
    ProjectOp and again separates the border from its consumer.

    N >> S with every index hit many times, so the atomic accumulation is real;
    integer dtypes keep scatter_add's parallel order-independent sum exact
    against eager.
    """

    def forward(
        self,
        base_a: Tensor,
        base_b: Tensor,
        index: Tensor,
        src: Tensor,
        x: Tensor,
        flags: Tensor,
    ) -> Tensor:
        base = base_a + base_b  # [S] dead intermediate -> clone elided
        acc = torch.ops.aten.scatter_add.default(base, 0, index, src)  # [S]
        # Consumer sized [S]: every operand is [S], none of the scatter's [N].
        limit = torch.minimum(
            torch.clamp(x, max=10000) + torch.clamp(acc, min=2000), x
        )
        picked = limit[index]  # [N], second use of 'index' -> alt Tensor copy
        sel = (src < picked) | flags  # [N]
        return torch.cumsum(sel.to(torch.int64), 0)


class SliceScatterDim1ViewTestPreproc(nn.Module):
    """Self-contained repro for the fused tw.slice_scatter dim=1 multi-row
    failure seen on the ROO batch=768 graph. Both outputs scatter into a dead
    intermediate base (base = x + y) so the inserted clone is elided and the
    scatter runs in place; dim=1; the number of rows (64) exceeds the inner
    extent (32) so any per-row failure is visible. Inputs are arange-based with
    distinct values so a mis-read is an obviously wrong value.

    out0 (control): the scattered src is a CONTIGUOUS [64, 32] intermediate.
    out1 (repro): the scattered src is a NON-CONTIGUOUS dim=1 slice (a view of a
    wider [64, 33] intermediate) fed through a clamp -- exactly the ROO
    slice_scatter(base, clamp(slice(x, dim=1, 0:C-1), -10, 10), dim=1, 0:C-1)
    pattern whose rows past the first came back as uninitialized garbage. The
    wide input is scaled into [-8, 8] so the clamp is a no-op and every value
    stays distinct.
    """

    def forward(
        self,
        base0_a: Tensor,
        base0_b: Tensor,
        src0_a: Tensor,
        src0_b: Tensor,
        base1_a: Tensor,
        base1_b: Tensor,
        wide_a: Tensor,
        wide_b: Tensor,
    ) -> tuple[Tensor, Tensor]:
        base0 = base0_a + base0_b  # [64, 33] dead intermediate -> clone elided
        src0 = src0_a + src0_b  # [64, 32] contiguous
        out0 = torch.ops.aten.slice_scatter.default(
            base0, src0, dim=1, start=0, end=32
        )

        base1 = base1_a + base1_b  # [64, 33] dead intermediate -> clone elided
        wide = wide_a + wide_b  # [64, 33]
        src1 = torch.ops.aten.slice.Tensor(wide, 1, 0, 32)  # [64, 32] non-contig
        src1 = torch.clamp(src1, -10.0, 10.0)
        out1 = torch.ops.aten.slice_scatter.default(
            base1,
            src1,
            dim=1,
            start=0,
            end=32,
        )
        return out0, out1


class SliceScatterOpenEndTestPreproc(nn.Module):
    """slice_scatter with aten's open-slice sentinel end (2**63-1), the form
    `base[:] = src` / `base[:, 2:] = src` functionalizes to. The ROO preproc
    label-weight build emits exactly this: select_scatter(dim=1, index=c)
    wrapped in slice_scatter(dim=0, start=0, end=2**63-1).

    Every other slice_scatter fixture here uses a small literal end, so this is
    the only one that fails when 'end' is narrowed to 32 bits: the sentinel
    reads back as -1, the slice length collapses to 0, and every element
    scatters onto index 0 along 'dim', leaving the output at its base value.

    out0: dim 0, the whole tensor (base[:] = src0), matching ROO.
    out1: dim 1 from column 2 on (base[:, 2:] = src1), so columns 0 and 1 must
    pass through untouched -- a scatter that silently writes nothing and one
    that overruns the slice are then both visible.
    """

    def forward(
        self,
        base0_a: Tensor,
        base0_b: Tensor,
        src0: Tensor,
        base1_a: Tensor,
        base1_b: Tensor,
        src1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        open_end = 9223372036854775807  # aten's "to the end" sentinel

        base0 = base0_a + base0_b  # dead intermediate -> clone elided
        out0 = torch.ops.aten.slice_scatter.default(
            base0, src0, dim=0, start=0, end=open_end, step=1
        )

        base1 = base1_a + base1_b
        out1 = torch.ops.aten.slice_scatter.default(
            base1, src1, dim=1, start=2, end=open_end, step=1
        )
        return out0, out1
