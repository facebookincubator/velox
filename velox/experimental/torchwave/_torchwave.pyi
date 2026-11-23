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

from typing import Any

class DelegateExecutor:
    def __init__(self, pt2_path: str) -> None: ...
    def run(self, inputs: list[Any]) -> list[Any]: ...
    def __repr__(self) -> str: ...

class TorchWaveModel:
    # 'inputs' are the graph's user inputs in order, one per forward argument;
    # pass None for an argument the exported graph carries as a None constant.
    # 'stream' is the cudaStream_t the caller's PyTorch work runs on (as
    # torch.cuda.Stream.cuda_stream gives it), 0 for the legacy default stream;
    # it must be the current stream. The GIL is released while the run blocks.
    def run(self, inputs: list[Any | None], stream: int = 0) -> list[Any]: ...
    # Like run() but reuses a single held execution frame across calls (weights
    # and constants set once, only inputs refilled). Single-threaded fast path.
    def run_reuse(self, inputs: list[Any | None], stream: int = 0) -> list[Any]: ...
    def __call__(self, inputs: list[Any | None], stream: int = 0) -> list[Any]: ...
    # The packed inputs load() rewrote after set_pack_inputs(True), one JSON
    # recipe each.
    def pack_recipes(self) -> list[str]: ...
    # Encodes the packed inputs among 'inputs'. Returns (input index, encoded
    # host tensor, stats) per packed input, and (input index, trimmed lengths,
    # {}) per lengths input of a windowed slot, the lengths being a tensor or,
    # for a scalar length, an int; put each value at its index before run().
    def encode_packed_inputs(
        self, inputs: list[Any | None], num_threads: int = ..., pinned: bool = ...
    ) -> list[tuple[int, Any, dict[str, int]]]: ...

def load(path: str, model_name: str = ...) -> TorchWaveModel: ...

class WaveConfig:
    block_size: int
    single_block_path_block_size: int
    all_standalone: bool
    num_standalone_threads: int
    # Read-only; set through set_is_cg, set_trace, set_enable_reuse and
    # set_kernel_cache_dir.
    @property
    def is_cg(self) -> bool | None: ...
    @property
    def trace(self) -> int: ...
    @property
    def enable_reuse(self) -> bool: ...
    @property
    def kernel_cache_dir(self) -> str: ...

def wave_config() -> WaveConfig: ...

# Per-op timing/perf report from the most recent run on this thread (populated
# when the executor's trace has the kTiming (16) bit set).
def last_perf_report() -> str: ...

# Set/get trace bits on the exact WaveConfig instance the executor reads (use
# these instead of wave_config().trace to avoid a duplicated inline-static
# WaveConfig instance across translation units).
def set_trace(trace: int) -> None: ...
def set_num_sms(num_sms: int) -> None: ...
def get_trace() -> int: ...
def set_is_cg(on: bool) -> None: ...
def set_enable_reuse(on: bool) -> None: ...
def set_kernel_cache_dir(dir: str) -> None: ...

# Rewrite packed inputs (split into one view per feature) in models loaded
# afterwards, so that they take encoded inputs.
def set_pack_inputs(on: bool) -> None: ...
def register_elementwise_op(
    qualified_name: str,
    elementwise_func_name: str,
    is_standalone: bool,
    attribute_args: list[str] = ...,
) -> None: ...
