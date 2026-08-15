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


class LogitTestPreproc(nn.Module):
    """Exercises aten.logit (inverse sigmoid), with and without eps.

    out_none: logit(x) with eps=None (no clamp); x is strictly in (0, 1).
    out_eps:  logit(y, eps=1e-3); y includes 0.0 and 1.0 to exercise the clamp
              to [eps, 1 - eps] (without it those would be +/- inf).
    out_fused: logit(x) + 1.0, so logit participates in an elementwise chain
               (confirms it lowers to a fused elementwise op, not a standalone).
    """

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out_none = torch.logit(x)
        out_eps = torch.logit(y, eps=1e-3)
        out_fused = torch.logit(x) + 1.0
        return out_none, out_eps, out_fused
