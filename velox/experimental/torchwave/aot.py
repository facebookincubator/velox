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

"""Whole-graph ahead-of-time packaging for TorchWave.

This is the whole-graph alternative to AOTInductor's
torch._inductor.aoti_compile_and_package: instead of lowering to a delegate
subgraph, the entire exported graph is packaged into a .pt2 archive and executed
by the TorchWave WaveGraphExecutor at load time (TorchWaveModel.load on the C++
side, or load() in this module). TorchWave handles supported ops as fused CUDA
kernels and falls back to nativert kernels for the rest, so any exported graph
can be packaged.
"""

from __future__ import annotations

import torch
from torch.export import ExportedProgram
from torch.export.pt2_archive._package import package_pt2
from torch.types import FileLike

DEFAULT_MODEL_NAME = "model"


def compile_and_package(
    exported_program: ExportedProgram,
    f: FileLike,
    model_name: str = DEFAULT_MODEL_NAME,
    decompose: bool = True,
) -> FileLike:
    """Package an ExportedProgram into a .pt2 that TorchWave runs whole-graph.

    Args:
        exported_program: the program to package, from torch.export.export().
        f: destination file path or file-like object for the .pt2 archive.
        model_name: name of the model inside the archive; TorchWaveModel.load
            selects it (or the first model when loaded with an empty name).
        decompose: when True, lower to the functional core-ATen opset that
            TorchWave expects via run_decompositions(); set False if the program
            is already in that form.

    Returns:
        The destination 'f', for convenience.
    """
    program = exported_program
    if decompose:
        program = program.run_decompositions(torch.export.default_decompositions())
    package_pt2(f, exported_programs={model_name: program})
    return f
