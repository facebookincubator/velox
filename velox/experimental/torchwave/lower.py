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

import types

import torch
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.pt2_archive._package import package_pt2
from torch.nativert.backends._lower_utils import get_new_ep_with_flat_inputs_outputs
from torch.nativert.backends._lowered_aoti_module import LoweredBackendModule
from torch.types import FileLike

TORCHWAVE_BACKEND_ID = "torchwave"


def lower_to_torchwave(
    exported_program: ExportedProgram, model_name: str
) -> ExportedProgram:
    """Lower an exported program to a torchwave delegate ExportedProgram.

    Takes any graph — torchwave handles supported ops natively and falls back
    to nativert kernels for the rest.
    """
    out_spec = exported_program.call_spec.out_spec
    flat_ep = get_new_ep_with_flat_inputs_outputs(exported_program)
    args, kwargs = exported_program.example_inputs

    lowered_module = LoweredBackendModule(
        flat_ep, TORCHWAVE_BACKEND_ID, module_name=model_name
    )

    def patched_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        flat_inputs, _ = pytree.tree_flatten((args, kwargs))
        flat_outputs = torch._higher_order_ops.executorch_call_delegate(
            self, *flat_inputs
        )
        if out_spec is not None and flat_outputs is not None:
            return pytree.tree_unflatten(flat_outputs, out_spec)
        else:
            return flat_outputs

    lowered_module.forward = types.MethodType(patched_forward, lowered_module)  # type: ignore[method-assign]

    delegate_ep = torch.export.export(lowered_module, args, kwargs)
    return delegate_ep


def package_nativert_with_torchwave_delegate(
    f: FileLike,
    model_name: str,
    original_ep: ExportedProgram,
    delegate_ep: ExportedProgram,
) -> None:
    """Package a pt2 archive for NativeRT with a torchwave delegate.

    The archive contains both the original model (used by the torchwave
    delegate executor to build the wave graph) and the delegate model
    (loaded by the nativert ModelRunner, contains executorch_call_delegate
    nodes).
    """
    package_pt2(
        f,
        exported_programs={
            model_name: original_ep,
            f"{model_name}-{TORCHWAVE_BACKEND_ID}": delegate_ep,
        },
    )
