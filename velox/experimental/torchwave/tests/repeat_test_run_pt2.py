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
from velox.experimental.torchwave.tests.repeat_test import RepeatTestPreproc


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write repeat_test.pt2 and repeat_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # A long 1-D vector (exercises the int64 element type) and float tensors of
    # rank 2 and 3, plus two identity inputs with distinct value ranges.
    vec = torch.arange(1, 5, dtype=torch.long)
    mat = torch.arange(12, dtype=torch.float).reshape(3, 4)
    cube = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
    id_vec = (torch.arange(5, dtype=torch.float) + 100).reshape(5)
    id_mat = (torch.arange(6, dtype=torch.float) + 200).reshape(2, 3)

    inputs = (vec, mat, cube, id_vec, id_mat)

    module = RepeatTestPreproc()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={tuple(r.shape)}, dtype={r.dtype}")

    results_path = os.path.join(output_dir, "repeat_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "repeat_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
