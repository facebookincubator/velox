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

"""A torch.compile backend that runs Dynamo-captured graphs on TorchWave.

The Dynamo-captured GraphModule is exported to an ExportedProgram, packaged into
a .pt2 archive with the same whole-graph packaging the ahead-of-time path uses
(aot.compile_and_package), and executed through the TorchWave GPU engine
(TorchWaveModel). The returned callable yields the flat tuple of output tensors,
which Dynamo unflattens back into the model's output structure.
"""

from __future__ import annotations

import logging
import os
import tempfile

import torch
from torch import Tensor

from velox.experimental.torchwave._torchwave import load, TorchWaveModel
from velox.experimental.torchwave.aot import compile_and_package

logger: logging.Logger = logging.getLogger(__name__)

TORCHWAVE_BACKEND_NAME = "torchwave"


def torchwave_backend(
    gm: torch.fx.GraphModule,
    example_inputs: list[Tensor],
):
    """Compile 'gm' for execution on TorchWave and return a callable graph.

    Under dynamic=True Dynamo passes the symbolic sizes as leading SymInt graph
    inputs, which torch.export cannot take. Rather than export here, export lazily
    per distinct runtime input shape: with concrete sizes those SymInt inputs are
    ordinary ints that torch.export bakes in as constants, leaving a tensor-only
    graph. One Dynamo graph can thus back several size-specialized TorchWave
    executors; this does not affect Dynamo's own recompile count.
    """
    executors: dict[tuple[object, ...], TorchWaveModel] = {}
    # Holds every compiled .pt2 for as long as this backend's executors live;
    # each reads weights from its archive lazily, so the files must outlive the
    # call that wrote them. Cleaned up when the closure is collected, which is
    # what keeps a long-running process from accumulating one file per shape.
    temp_dir = tempfile.TemporaryDirectory(prefix="torchwave_backend_")

    def _key(args: tuple[object, ...]) -> tuple[object, ...]:
        """Cache key covering everything export specializes on.

        Shape and dtype for a tensor, and the VALUE of every non-tensor: export
        bakes a scalar into the graph as a literal, so keying on shapes alone
        would hand a call with a different scalar the executor built for the
        first one, which returns the first one's answer. A caller that sets
        torch._dynamo.config.specialize_float would not reach that case, but
        the backend cannot assume it. Same discipline as multi_signature.
        """
        return tuple(
            (
                ("t", tuple(a.shape), str(a.dtype))
                if isinstance(a, Tensor)
                else ("s", type(a).__name__, a)
            )
            for a in args
        )

    def compiled(*args: object) -> tuple[Tensor, ...]:
        # A new key re-exports; that is a backend detail and does not count as a
        # Dynamo recompile.
        key = _key(args)
        model = executors.get(key)
        if model is None:
            exported = torch.export.export(gm, tuple(args))
            pt2_path = os.path.join(temp_dir.name, f"model_{len(executors)}.pt2")
            compile_and_package(exported, pt2_path)
            logger.info("torchwave backend: wrote temporary pt2 to %s", pt2_path)
            model = load(pt2_path)
            executors[key] = model
        # Pass all inputs (symbolic sizes are concrete ints here, plus tensors)
        # so the count matches the exported graph's user inputs.
        return tuple(model.run(list(args)))

    return compiled


def register() -> None:
    """Register torchwave_backend under the name 'torchwave' with Dynamo."""
    torch._dynamo.register_backend(
        compiler_fn=torchwave_backend, name=TORCHWAVE_BACKEND_NAME
    )
