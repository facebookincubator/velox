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


class MaskedSelectEmptyTestPreproc(nn.Module):
    """masked_select over a dynamically-empty slice.

    'end' is a runtime scalar (end.item()), so slicing data/flags to [:end]
    produces tensors whose length is data-dependent: the exported graph reserves
    the full tensor length as an upper bound, but at runtime end=0 makes the
    sliced flags (and data) empty. This is the "reserved capacity > 0, actual
    length 0" case that masked_select must report as an empty ([0]) output rather
    than leaking its reserved size. A statically empty input would not reproduce
    it, because the graph would then reserve length 0.

    Inputs: data (longs), flags (bool, same length), end (0-d long = 0).
    Output: torch.masked_select(data[:end], flags[:end]) -> empty long tensor.
    """

    def forward(self, data: Tensor, flags: Tensor, end: Tensor) -> Tensor:
        k = end.item()
        # end.item() is statically typed bool|int|float, but at runtime it is a
        # SymInt (0-d Long tensor); _check_is_size marks it as a size.
        torch._check_is_size(k)  # pyre-ignore[6]
        torch._check(k <= flags.shape[0])
        return torch.masked_select(data[:k], flags[:k])
