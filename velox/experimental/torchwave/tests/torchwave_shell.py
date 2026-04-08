# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import code

import torch
from velox.experimental.torchwave._torchwave import DelegateExecutor
from velox.experimental.torchwave.tests.element_test import ElementTestPreproc


def main() -> None:
    print("TorchWave interactive shell")
    print("Available: torch, ElementTestPreproc, DelegateExecutor")
    print("Example:")
    print("  module = ElementTestPreproc()")
    print("  inputs = tuple(torch.arange(1, 10001, dtype=torch.long) for _ in range(6))")
    print("  outputs = module(*inputs)")
    code.interact(local={
        **globals(),
        "torch": torch,
        "ElementTestPreproc": ElementTestPreproc,
        "DelegateExecutor": DelegateExecutor,
    })


if __name__ == "__main__":
    main()
