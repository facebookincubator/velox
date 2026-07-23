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

from __future__ import annotations

import argparse
import os

import torch
from velox.experimental.torchwave.tests.masked_select_compose_test import (
    MaskedSelectComposeTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument("--output_dir", default=default_dir)
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # A 4K-element arange: the masked_selects keep the multiples of 10 then 20.
    stuff = torch.arange(0, 4096, dtype=torch.long)

    module = MaskedSelectComposeTestPreproc()
    result = module(stuff)
    print(f"Eager result: shape={result.shape}, first 10={result[:10]}")

    results_path = os.path.join(output_dir, "masked_select_compose_test_results.pt")
    torch.save([result], results_path)
    print(f"Saved results to {results_path}")

    with torch.no_grad():
        exported_program = torch.export.export(module, (stuff,), strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "masked_select_compose_test.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"Saved .pt2 to {pt2_path} ({os.path.getsize(pt2_path)} bytes)")


if __name__ == "__main__":
    main()
