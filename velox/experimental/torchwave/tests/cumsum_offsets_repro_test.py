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


def _offsets_chain(x: Tensor, off: int) -> Tensor:
    # The cumsum reads a select-view of a wave-produced cast, then feeds an
    # exclusive-prefix cat([zeros[1], cumsum[:-1]]) (the ads "inner offsets"
    # motif).  Two things this exercises that a plain cat(cumsum) does not:
    #   1. the cat element is slice(cumsum) -- a VIEW -- so the cat must still
    #      emit the cumsum compute under the view (not just copy a stale buffer);
    #   2. the cat is mixed-dtype (float zeros + int64 cumsum), so the element
    #      must be value-converted, not bit-copied.
    base = (x + off).to(torch.int32)
    lengths = base[:, 0, 1]
    cs = torch.cumsum(lengths, 0, dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.float32), cs[:-1]])
    # A second select-view consumer of `base` keeps the cast multi-consumer, so
    # it materializes as a standalone the cumsum reads cross-kernel (mirroring
    # the ads select-view of a multi-consumer _to_copy).
    other = base[:, 1, 0].to(torch.float32)
    return offsets + other


class CumsumOffsetsReproPreproc(nn.Module):
    """Concurrent inner-offset chains: cumsum(select(multi-consumer cast)) fed
    into an exclusive-prefix cat([zeros[1], cumsum[:-1]]).  Reproduces, without
    the 24GB ads reference frame, two deterministic cat-codegen bugs: dropping
    the cumsum compute under a view cat-element, and bit-copying (vs converting)
    a mixed-dtype cat element."""

    def forward(self, x: Tensor) -> Tensor:
        outs = [_offsets_chain(x, off) for off in range(4)]
        return torch.cat(outs)
