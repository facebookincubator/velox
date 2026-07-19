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


class MaskedSelectComposeTestPreproc(nn.Module):
    """Composed masked_selects feeding an elementwise add.

    Exercises a masked_select (whose length is set on device) as the input to an
    elementwise op. The result is o = masked_select(masked_select(stuff, f1), f2)
    * 2 + masked_select(stuff, composed), where applying f1 then f2 selects the
    same elements as composed, so both arguments of the add have equal length.

    With stuff = arange(4096): f1 = stuff % 10 == 0 keeps the multiples of 10;
    f2 = inner % 20 == 0 then keeps the multiples of 20; composed = stuff % 20 ==
    0 selects those same multiples of 20 directly. Both arguments are therefore
    [0, 20, 40, ..., 4080] and o = 3 * [0, 20, ..., 4080].
    """

    def forward(self, stuff: Tensor) -> Tensor:
        f1 = (stuff % 10) == 0
        inner = stuff.masked_select(f1)
        f2 = (inner % 20) == 0
        outer = inner.masked_select(f2)
        composed = (stuff % 20) == 0
        other = stuff.masked_select(composed)
        return outer * 2 + other
