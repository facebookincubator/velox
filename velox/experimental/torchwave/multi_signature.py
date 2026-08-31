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

"""Signature-keyed executor cache for the TorchWave whole-graph entry point.

One compiled TorchWave executor handles a single fixed graph. It runs any input
SIZE at runtime (the wave graph carries dynamic size expressions) but is tied to
the graph's dtypes, ranks, and None-vs-tensor structure. Feeding a batch whose
dtype, rank, or None structure differs needs a different graph.

MultiSignatureModel is the TorchWave analog of Dynamo's per-code-object guard
cache: it wraps an eager nn.Module and, on first sight of each distinct input
SIGNATURE, exports and compiles a TorchWave executor for that batch, then reuses
it for every later batch with the same signature. A batch that differs only in
size reuses the existing executor and does not recompile.
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.export import Dim

from velox.experimental.torchwave._torchwave import load, TorchWaveModel
from velox.experimental.torchwave.aot import compile_and_package

# The signature key: the flattened-input pytree structure (as a string) plus a
# per-leaf descriptor. Structure captures None-vs-tensor and container changes;
# the leaf descriptors capture a tensor's dtype and rank and a scalar's value.
# Tensor SIZE is deliberately excluded -- that is the whole point, since the
# wave graph carries dynamic size expressions -- but a scalar's value is not,
# because export bakes it in as a literal.
SignatureKey = tuple[str, tuple[tuple[object, ...], ...]]


def _leaf_signature(leaf: object) -> tuple[object, ...]:
    """Signature of one flattened leaf: None, a tensor's (dtype, rank), or a
    non-tensor constant's type and value."""
    if leaf is None:
        return ("none",)
    if isinstance(leaf, Tensor):
        return ("tensor", str(leaf.dtype), leaf.dim())
    # The value, not only the type: _compile exports with these arguments and
    # torch.export bakes a scalar into the graph as a literal. Keying on the
    # type alone would hand a batch with a different scalar the executor
    # compiled for the first one, which returns the first one's answer.
    return ("scalar", type(leaf).__name__, leaf)


def _signature(args: tuple[object, ...]) -> SignatureKey:
    leaves, spec = pytree.tree_flatten(args)
    return (str(spec), tuple(_leaf_signature(leaf) for leaf in leaves))


def _dynamic_shapes(args: tuple[object, ...]) -> tuple[object, ...]:
    """Mark every size dim of every tensor arg dynamic so batches that differ
    only in size share one executor. Rank stays fixed (it is part of the key),
    and Dim.AUTO specializes any dim the graph cannot make dynamic (e.g. a
    broadcast dim of size 1)."""

    def per_arg(arg: object) -> object:
        if isinstance(arg, Tensor) and arg.dim() > 0:
            return {dim: Dim.AUTO for dim in range(arg.dim())}
        return None

    return tuple(per_arg(arg) for arg in args)


class MultiSignatureModel:
    """Wraps an eager nn.Module with a signature-keyed TorchWave executor cache.

    Call it like the eager module; it returns the flat list of output tensors
    produced by the TorchWave executor for the matching signature.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self._cache: dict[SignatureKey, TorchWaveModel] = {}
        # Holds every compiled .pt2 for the wrapper's lifetime; each executor
        # reads weights from its archive lazily, so the files must outlive it.
        self._temp_dir = tempfile.TemporaryDirectory(prefix="torchwave_multisig_")

    @property
    def num_compiles(self) -> int:
        """Number of distinct signatures compiled so far (one executor each)."""
        return len(self._cache)

    def __call__(self, *args: object) -> list[Tensor]:
        key = _signature(args)
        executor = self._cache.get(key)
        if executor is None:
            executor = self._compile(args)
            self._cache[key] = executor
        # torch.export keeps one user input per forward argument: tensors are
        # real inputs, while None and scalar arguments become dead placeholders
        # (their value is baked into the ops as a literal). Pass every leaf in
        # order so the count matches the graph's user inputs; the executor binds
        # tensors and ignores the dead placeholder slots.
        return executor.run(pytree.tree_leaves(args))

    def _compile(self, args: tuple[object, ...]) -> TorchWaveModel:
        with torch.no_grad():
            exported = torch.export.export(
                self._model,
                args,
                strict=False,
                dynamic_shapes=_dynamic_shapes(args),
            )
        pt2_path = os.path.join(self._temp_dir.name, f"model_{len(self._cache)}.pt2")
        compile_and_package(exported, pt2_path)
        return load(pt2_path)
