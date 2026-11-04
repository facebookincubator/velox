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

from torch import nn, Tensor


class ContiguousTest(nn.Module):
    """aten.contiguous on every layout the fused implementation distinguishes.

    Each result is flattened with view(-1), which reads it in row-major order,
    so a contiguous that returned its strided input, or wrote the wrong
    layout, shows up as misplaced elements.

    Inputs: x (70x45 float), big (300x257 float), z (40x33 int64),
    w (12x10 float), y (6x5x4 float).
    Outputs:
        transposed: x.t(), a 2D transpose through shared-memory tiles, with
            partial tiles on both edges.
        transposed_big: big.t(), many tiles spread over many blocks.
        transposed_long: z.t(), tiles of 8-byte elements.
        known: (w + 1), known contiguous, so the contiguous is elided.
        permuted: a rank-3 permute, copied as a clone copies.
        sliced: x[:, 3:40], 2D with unit row stride but a row pitch.
        column: x[:, 5], a strided rank-1 tensor.
        input: x itself, contiguous only at run time.
    """

    def forward(
        self,
        x: Tensor,
        big: Tensor,
        z: Tensor,
        w: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        transposed = x.t().contiguous().view(-1)
        transposed_big = big.t().contiguous().view(-1)
        transposed_long = z.t().contiguous().view(-1)
        known = (w + 1).contiguous().view(-1)
        permuted = y.permute(2, 0, 1).contiguous().view(-1)
        sliced = x[:, 3:40].contiguous().view(-1)
        column = x[:, 5].contiguous()
        whole = x.contiguous().view(-1)
        return (
            transposed,
            transposed_big,
            transposed_long,
            known,
            permuted,
            sliced,
            column,
            whole,
        )
