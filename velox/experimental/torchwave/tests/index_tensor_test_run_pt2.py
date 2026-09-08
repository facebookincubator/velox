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
from velox.experimental.torchwave.tests.index_tensor_test import IndexTensorTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write index_tensor_test.pt2 and "
        "index_tensor_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    x = torch.arange(4 * 5 * 6, dtype=torch.float).reshape(4, 5, 6)
    # Single index per case, each selecting one dim -> fused index_select.
    sel0 = torch.tensor([3, 1, 2], dtype=torch.long)  # dim 0, in [0, 4)
    sel1 = torch.tensor([4, 0, 2, 2], dtype=torch.int32)  # dim 1, in [0, 5)
    sel2 = torch.tensor([5, 1, 0, 3, 5], dtype=torch.long)  # dim 2, in [0, 6)
    # Two separated indices (dim 0 and dim 2) -> eager fallback.
    sep0 = torch.tensor([1, 3], dtype=torch.long)  # in [0, 4)
    sep2 = torch.tensor([0, 5], dtype=torch.long)  # in [0, 6)
    # 1-D source + 1-D boolean mask -> masked_select.
    x1d = torch.arange(20, dtype=torch.float)
    mask = (torch.arange(20) % 3) != 0  # boolean, mixed true/false

    inputs = (x, sel0, sel1, sel2, sep0, sep2, x1d, mask)

    module = IndexTensorTest()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={tuple(r.shape)}, dtype={r.dtype}")

    results_path = os.path.join(output_dir, "index_tensor_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "index_tensor_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
