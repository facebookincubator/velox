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


class FoldTestPreproc(nn.Module):
    """Tests constant folding, redundant views, and mixed foldable/dynamic ops.

    Has weight-only subgraphs that should fold to constants, redundant
    view(-1) chains between masked_selects, and constant masks/inputs.

    Inputs: x (1D long, 10000 elements)
    Weights: w1 (1D long, 10000), w2 (1D long, 10000)
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("w1", torch.arange(0, 10000, dtype=torch.long))
        self.register_buffer("w2", torch.arange(10000, 20000, dtype=torch.long))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # --- Foldable subgraphs (depend only on weights) ---

        # Folds to a constant tensor: element-wise on weights only.
        # pyrefly: ignore [unsupported-operation]
        folded_sum = self.w1 + self.w2

        # Folds to a constant scalar via reduction.
        # pyrefly: ignore [not-callable]
        folded_scalar = self.w1.sum()

        # Folds to a constant mask: w1 % 10 < 9 (90% true).
        # pyrefly: ignore [unsupported-operation]
        folded_mask = self.w1 % 10 < 9

        # Folds to a constant tensor: masked_select with constant input and mask.
        # pyrefly: ignore [bad-argument-type]
        folded_selected = torch.masked_select(self.w2, folded_mask)

        # --- Dynamic ops with redundant views ---

        # Chain 1: view, view, masked_select with folded mask.
        t1 = x.view(-1)
        t2 = t1.view(-1)
        # pyrefly: ignore [bad-argument-type]
        s1 = torch.masked_select(t2, folded_mask)

        # Chain 2: view, view, masked_select with dynamic mask.
        t3 = s1.view(-1)
        t4 = t3.view(-1)
        dynamic_mask = t4 % 3 != 0
        s2 = torch.masked_select(t4, dynamic_mask)

        # Chain 3: more views, then use the folded constant tensor.
        t5 = s2.view(-1)
        t6 = t5.view(-1)
        # Use folded_scalar (constant) as an addend.
        o1 = t6 + folded_scalar

        # Use folded_sum (constant tensor) with dynamic input.
        o2 = x + folded_sum

        # Use folded_selected (constant tensor from masked_select on weights).
        o3 = folded_selected + 1

        # Another foldable scalar: product of weight shapes.
        # pyrefly: ignore [bad-index]
        folded_numel = torch.tensor(self.w1.shape[0] * self.w2.shape[0])
        o4 = x + folded_numel

        return o1, o2, o3, o4
