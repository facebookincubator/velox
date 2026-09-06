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
    def run(self, inputs: list[Any | None]) -> list[Any]: ...
    # Like run() but reuses a single held execution frame across calls (weights
    # and constants set once, only inputs refilled). Single-threaded fast path.
    def run_reuse(self, inputs: list[Any | None]) -> list[Any]: ...
    def __call__(self, inputs: list[Any | None]) -> list[Any]: ...

def load(path: str, model_name: str = ...) -> TorchWaveModel: ...

class WaveConfig:
    block_size: int
    single_block_path_block_size: int
    all_standalone: bool
    num_standalone_threads: int
    is_cg: bool | None
    trace: int
    enable_reuse: bool
    kernel_cache_dir: str

def wave_config() -> WaveConfig: ...

# Per-op timing/perf report from the most recent run on this thread (populated
# when the executor's trace has the kTiming (16) bit set).
def last_perf_report() -> str: ...

# Set/get trace bits on the exact WaveConfig instance the executor reads (use
# these instead of wave_config().trace to avoid a duplicated inline-static
# WaveConfig instance across translation units).
def set_trace(trace: int) -> None: ...
def get_trace() -> int: ...
def register_elementwise_op(
    qualified_name: str,
    elementwise_func_name: str,
    is_standalone: bool,
    attribute_args: list[str] = ...,
) -> None: ...
