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


class CopyInPlaceLayoutPreproc(nn.Module):
    """In-place `copy_` for each combination of contiguous and strided source
    and destination. The graph is exported without functionalization, so every
    write stays an `aten.copy_` instead of select_scatter + copy. Each source is
    an input or a view of one, so it is read from memory at its own layout
    rather than handed over in a register.

      a: contiguous <- contiguous   whole-tensor copy
      b: contiguous <- strided      source is a transpose
      c: strided    <- contiguous   destination is a column view
      d: strided    <- strided      column to column, chained over four columns
                                    in reverse order, as in the ROO preproc
    """

    def forward(
        self, dst: Tensor, src: Tensor, square: Tensor, vec: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        a = dst * 2.0
        a.copy_(src)

        b = square * 2.0
        b.copy_(square.t())

        c = dst * 2.0
        c[:, 2].copy_(vec)

        d = dst * 2.0
        d[:, 0].copy_(src[:, 3])
        d[:, 1].copy_(src[:, 2])
        d[:, 2].copy_(src[:, 1])
        d[:, 3].copy_(src[:, 0])

        return a, b, c, d


class CopyInPlaceConvertPreproc(nn.Module):
    """In-place `copy_` where the source broadcasts or converts, and where it
    reads a snapshot of the buffer it writes.

      a: [R, C] <- [1, C]   size-1 dim broadcast over rows
      b: column <- expand   an explicit aten.expand of a one-element source
      c: column <- int64    dtype conversion into a strided destination
      d: shift by one       `out[1:].copy_(out[:-1].clone())`; the clone is a
                            required snapshot because `out` is overwritten
    """

    def forward(
        self, dst: Tensor, row: Tensor, one: Tensor, ints: Tensor, base: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rows = dst.size(0)

        a = dst * 2.0
        a.copy_(row)

        b = dst * 2.0
        b[:, 0].copy_(one.expand(rows))

        c = dst * 2.0
        c[:, 1].copy_(ints[:, 1])

        d = base * 2.0
        d[1:].copy_(d[:-1].clone())

        return a, b, c, d


def make_inputs() -> dict[str, tuple[Tensor, ...]]:
    """Inputs for each module, shared by the exporter and the C++ test data."""
    rows, cols = 26, 8
    dst = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    src = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) * -1.0
    square = torch.arange(cols * cols, dtype=torch.float32).reshape(cols, cols)
    vec = torch.arange(rows, dtype=torch.float32) + 500.0
    row = torch.arange(cols, dtype=torch.float32).reshape(1, cols) + 100.0
    one = torch.tensor([7.0], dtype=torch.float32)
    ints = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
    base = torch.arange(1, 4097, dtype=torch.float32)
    return {
        "copy_inplace_layout_test": (dst, src, square, vec),
        "copy_inplace_convert_test": (dst, row, one, ints, base),
    }


def make_modules() -> dict[str, nn.Module]:
    return {
        "copy_inplace_layout_test": CopyInPlaceLayoutPreproc(),
        "copy_inplace_convert_test": CopyInPlaceConvertPreproc(),
    }
