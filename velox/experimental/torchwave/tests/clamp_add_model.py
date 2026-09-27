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

"""Shared model and batch matrix for the TorchWave recompile tests.

clamp(x, low, high) + y over a matrix that forces a fresh compile for every
distinct input signature: x rank in {1, 2, 3}, low/high each present or absent
(None), and y either the same shape as x or a lower-rank tensor that broadcasts
to it. Each combination is a distinct graph; two batches that differ only in
size must share one compile. Both the ahead-of-time and the dynamo test import
this so they run the identical batches against the identical eager reference.
"""

from __future__ import annotations

import torch
from torch import Tensor

RANKS: tuple[int, ...] = (1, 2, 3)
# Presence of (low, high) for clamp: neither, min-only, max-only, both.
LOW_HIGH_PRESENCE: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)
Y_BROADCAST: tuple[bool, ...] = (False, True)

# (x, low, high, y); low/high are scalar bounds, None when absent. Scalars keep
# clamp on the fused aten.clamp.default path (TorchWave does not fuse the tensor
# overload aten.clamp.Tensor).
Batch = tuple[Tensor, float | None, float | None, Tensor]


class ClampAddModel(torch.nn.Module):
    """out = clamp(x, low, high) + y. With both bounds absent there is no clamp
    node at all, which is itself a distinct graph."""

    def forward(
        self,
        x: Tensor,
        low: float | None,
        high: float | None,
        y: Tensor,
    ) -> Tensor:
        if low is None and high is None:
            return x + y
        return torch.clamp(x, low, high) + y


def _shape(rank: int, lead: int) -> tuple[int, ...]:
    if rank == 1:
        return (lead,)
    if rank == 2:
        return (lead, 5)
    return (lead, 3, 5)


def _make_batch(
    rank: int,
    has_low: bool,
    has_high: bool,
    y_broadcast: bool,
    lead: int,
) -> Batch:
    shape = _shape(rank, lead)
    x = torch.randn(shape)
    low = -0.5 if has_low else None
    high = 0.5 if has_high else None
    # A broadcast y drops the leading dim, so its rank is one less than x's and
    # it still trailing-aligns; rank 1 collapses to a 0-d scalar.
    y = torch.randn(shape[1:]) if y_broadcast else torch.randn(shape)
    return (x, low, high, y)


def build_matrix() -> tuple[list[Batch], int, Batch]:
    """One batch per distinct signature, plus a same-signature/different-size
    batch that must reuse an existing compile. Returns (batches,
    num_distinct_signatures, size_variant)."""
    torch.manual_seed(0)
    batches: list[Batch] = []
    for rank in RANKS:
        for has_low, has_high in LOW_HIGH_PRESENCE:
            for y_broadcast in Y_BROADCAST:
                batches.append(
                    _make_batch(rank, has_low, has_high, y_broadcast, lead=4)
                )
    # Same signature as the rank-2, both-bounds-present, non-broadcast batch, but
    # with a different leading size.
    size_variant = _make_batch(2, True, True, False, lead=9)
    return batches, len(batches), size_variant
