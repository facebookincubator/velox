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

"""Signature-keyed recompile behavior of the TorchWave ahead-of-time path.

Drives MultiSignatureModel over the clamp(x, low, high) + y matrix and checks
that it compiles one executor per distinct input signature, reuses it for a new
size, and matches eager on every batch.
"""

from __future__ import annotations

import unittest

import torch

from velox.experimental.torchwave.multi_signature import MultiSignatureModel
from velox.experimental.torchwave.tests.clamp_add_model import (
    build_matrix,
    ClampAddModel,
)


class MultiSignatureAotiTest(unittest.TestCase):
    def test_recompile_matches_signatures(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("TorchWave requires CUDA")

        model = ClampAddModel()
        wrapper = MultiSignatureModel(model)
        batches, num_signatures, size_variant = build_matrix()

        for batch in batches:
            expected = model(*batch)
            actual = wrapper(*batch)
            torch.testing.assert_close(actual[0].cpu(), expected, rtol=1e-4, atol=1e-4)

        # A fresh size for an already-seen signature must not add a compile.
        compiles_before = wrapper.num_compiles
        expected = model(*size_variant)
        actual = wrapper(*size_variant)
        torch.testing.assert_close(actual[0].cpu(), expected, rtol=1e-4, atol=1e-4)

        self.assertEqual(wrapper.num_compiles, compiles_before)
        self.assertEqual(wrapper.num_compiles, num_signatures)
