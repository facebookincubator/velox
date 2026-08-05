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
from velox.experimental.torchwave.tests.index_select_test import IndexSelectTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write index_select_test.pt2 and "
        "index_select_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Part 1: a 12x16x20 tensor sliced along each dim with 5 indices.
    base = torch.arange(12 * 16 * 20, dtype=torch.float).reshape(12, 16, 20)
    idx0 = torch.tensor([11, 0, 5, 3, 8], dtype=torch.long)
    idx1 = torch.tensor([15, 1, 7, 0, 10], dtype=torch.int32)
    idx2 = torch.tensor([19, 2, 0, 13, 6], dtype=torch.long)

    # Part 2/3: three 10x10x10 cubes with distinct value ranges plus a 10x10
    # matrix, sliced with 1-element indices so the slices broadcast.
    a = torch.arange(1000, dtype=torch.float).reshape(10, 10, 10)
    b = (torch.arange(1000, dtype=torch.float) + 1000).reshape(10, 10, 10)
    c = (torch.arange(1000, dtype=torch.float) + 2000).reshape(10, 10, 10)
    mat2d = (torch.arange(100, dtype=torch.float) + 3000).reshape(10, 10)
    one0 = torch.tensor([3], dtype=torch.long)
    one1 = torch.tensor([7], dtype=torch.int32)
    one2 = torch.tensor([2], dtype=torch.long)

    inputs = (base, idx0, idx1, idx2, a, b, c, one0, one1, one2, mat2d)

    module = IndexSelectTest()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={tuple(r.shape)}, dtype={r.dtype}")

    results_path = os.path.join(output_dir, "index_select_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "index_select_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
