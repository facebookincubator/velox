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

class WaveConfig:
    block_size: int
    single_block_path_block_size: int
    all_standalone: bool
    num_standalone_threads: int

def wave_config() -> WaveConfig: ...
def register_elementwise_op(
    qualified_name: str,
    elementwise_func_name: str,
    is_standalone: bool,
    attribute_args: list[str] = ...,
) -> None: ...
