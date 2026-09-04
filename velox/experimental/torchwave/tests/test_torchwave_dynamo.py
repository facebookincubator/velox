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

"""Signature-keyed recompile behavior of the TorchWave dynamo backend.

Drives torch.compile(backend="torchwave", dynamic=True) over the same
clamp(x, low, high) + y matrix as the ahead-of-time test and checks that Dynamo
compiles one graph per distinct input signature, reuses it for a new size, and
matches eager on every batch.
"""

from __future__ import annotations

import gc
import unittest

import torch
from torch import Tensor
from torch._dynamo.utils import counters

from velox.experimental.torchwave.dynamo_backend import (
    register,
    TORCHWAVE_BACKEND_NAME,
)
from velox.experimental.torchwave.tests.clamp_add_model import (
    build_matrix,
    ClampAddModel,
)


def _to_cuda(batch: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(arg.to("cuda") if isinstance(arg, Tensor) else arg for arg in batch)


# torch._dynamo.config fields this test overrides. They are process-global, so
# they are saved in setUp and put back in tearDown; left mutated they make any
# later test in the same process depend on the order it ran in.
_DYNAMO_CONFIG_KEYS = (
    "cache_size_limit",
    "accumulated_cache_size_limit",
    "specialize_float",
)


class TorchwaveDynamoTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dynamo_config = {
            key: getattr(torch._dynamo.config, key) for key in _DYNAMO_CONFIG_KEYS
        }

    def tearDown(self) -> None:
        for key, value in self._dynamo_config.items():
            setattr(torch._dynamo.config, key, value)
        # Drop the compiled functions so their cached TorchWave executors release
        # their kernels before process exit; otherwise the static kernel cache is
        # destroyed with entries still pinned and aborts.
        torch._dynamo.reset()
        gc.collect()

    def test_recompile_matches_signatures(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("TorchWave requires CUDA")

        register()
        torch._dynamo.reset()
        # The matrix has more distinct signatures than the default recompile
        # cache limit (8); raise it so every signature compiles instead of
        # falling back to eager once the limit is hit.
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.accumulated_cache_size_limit = 256
        # Bake scalar clamp bounds into the graph as literals (like the aot path)
        # instead of promoting them to tensor inputs with a runtime .item(), which
        # TorchWave cannot fuse into aten.clamp.default.
        torch._dynamo.config.specialize_float = True
        counters.clear()

        model = ClampAddModel()
        compiled = torch.compile(model, backend=TORCHWAVE_BACKEND_NAME, dynamic=True)
        batches, num_signatures, size_variant = build_matrix()

        for batch in batches:
            expected = model(*batch)
            with torch.no_grad():
                actual = compiled(*_to_cuda(batch))
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)

        # A fresh size for an already-seen signature must not add a graph.
        graphs_before = counters["stats"]["unique_graphs"]
        expected = model(*size_variant)
        with torch.no_grad():
            actual = compiled(*_to_cuda(size_variant))
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)

        self.assertEqual(counters["stats"]["unique_graphs"], graphs_before)
        self.assertEqual(counters["stats"]["unique_graphs"], num_signatures)
