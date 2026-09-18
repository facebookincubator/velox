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


class CopyColumnPreproc(nn.Module):
    """Column writes through `.copy_`, as they appear in the ROO preproc graph.

    `dst[:, k].copy_(v)` functionalizes to

        slice(dst, 0, 0, MAX)
        clone(that)
        copy(select(that, 1, k), v)
        select_scatter(clone, copy_result, 1, k)
        slice_scatter(dst, select_scatter_result, 0, 0, MAX)

    so the copy is the innermost producer and the only non-fusable node in the
    chain. The writes are chained (each reads the previous result), which is
    what makes the ROO graph spend a step per column.
    """

    def forward(self, dst: Tensor, src: Tensor) -> Tensor:
        out = dst * 2.0
        out[:, 0].copy_(src[:, 3])
        out[:, 1].copy_(src[:, 2])
        out[:, 2].copy_(src[:, 1])
        out[:, 3].copy_(src[:, 0])
        return out


class CopyBroadcastPreproc(nn.Module):
    """`copy` where the source broadcasts and/or converts to the destination.

    The destination supplies the shape and the dtype; the source supplies only
    values. Each case is returned separately so a failure names the case.

      a: [R, C] <- [1, C]   size-1 dim broadcast over rows (stride 0 on dim 0)
      b: [R]    <- [1]      size-1 broadcast of a one-element source
      c: [R]    <- int64    dtype conversion, no broadcast
      d: [R]    <- expand   an explicit aten.expand feeding the copy, which is
                            the shape the ROO graph produces
      e: [R, C] <- [C]      rank-broadcast: a lower-rank source right-aligned
    """

    def forward(
        self, dst: Tensor, row: Tensor, one: Tensor, ints: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        rows = dst.size(0)

        a = dst * 2.0
        a.copy_(row)

        b = dst * 2.0
        b[:, 0].copy_(one)

        c = dst * 2.0
        c[:, 1].copy_(ints[:, 1])

        d = dst * 2.0
        d[:, 2].copy_(one.expand(rows))

        e = dst * 2.0
        e.copy_(row.view(-1))

        return a, b, c, d, e


class CopyOverlapPreproc(nn.Module):
    """A `copy` whose source aliases the buffer the enclosing scatter writes.

    `out[1:]` and `out[:-1]` are views of one buffer shifted by one element, so
    the write and the read overlap at a non-zero offset. The `.clone()` makes
    the eager semantics well defined (and is what eager requires: `copy_` on
    partially overlapping tensors raises), but `out`'s previous value is dead
    afterwards, so clone elision is free to drop the snapshot and let the
    scatter write `out` in place.

    Once that happens a register-valued copy is wrong: the copy's read and the
    scatter's write land in the same fused loop over one buffer, with no
    ordering between lanes, so lane i may read an element lane i-1 has already
    overwritten. The correct result is a shift; the hazard collapses the tail
    to a run of the first element. The copy has to materialize its output so
    the whole read completes (behind a barrier) before the write starts.

    Also covers the 2-D column form, where the overlap is strided rather than
    contiguous.
    """

    def forward(self, base: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        out = base * 2.0
        out[1:].copy_(out[:-1].clone())

        g = grid * 2.0
        g[:, 1:].copy_(g[:, :-1].clone())

        return out, g


def make_inputs() -> dict[str, tuple[Tensor, ...]]:
    """Inputs for each module, shared by the exporter and the C++ test data."""
    torch.manual_seed(0)
    rows, cols = 26, 8
    dst = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    src = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) * -1.0
    row = torch.arange(cols, dtype=torch.float32).reshape(1, cols) + 100.0
    one = torch.tensor([7.0], dtype=torch.float32)
    ints = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
    # Wide enough for the overlapping shift to span many blocks. The hazard is
    # a race between lanes, so a wrong lowering does not reliably produce wrong
    # numbers -- the plan assertion in copyOverlapTest is what catches it.
    over_rows, over_cols = 512, 16
    base = torch.arange(1, over_rows * over_cols + 1, dtype=torch.float32)
    grid = torch.arange(over_rows * over_cols, dtype=torch.float32).reshape(
        over_rows, over_cols
    )
    return {
        "copy_column_test": (dst, src),
        "copy_broadcast_test": (dst, row, one, ints),
        "copy_overlap_test": (base, grid),
    }


def make_modules() -> dict[str, nn.Module]:
    return {
        "copy_column_test": CopyColumnPreproc(),
        "copy_broadcast_test": CopyBroadcastPreproc(),
        "copy_overlap_test": CopyOverlapPreproc(),
    }
