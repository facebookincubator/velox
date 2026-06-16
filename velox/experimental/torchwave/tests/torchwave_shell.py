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

import code

import torch
from velox.experimental.torchwave.tests.element_test import ElementTestPreproc


def main() -> None:
    print("TorchWave interactive shell")
    print("Available: torch, ElementTestPreproc")
    print("Example:")
    print("  module = ElementTestPreproc()")
    print(
        "  inputs = tuple(torch.arange(1, 10001, dtype=torch.long) for _ in range(6))"
    )
    print("  outputs = module(*inputs)")
    code.interact(
        local={
            **globals(),
            "torch": torch,
            "ElementTestPreproc": ElementTestPreproc,
        }
    )


if __name__ == "__main__":
    main()
