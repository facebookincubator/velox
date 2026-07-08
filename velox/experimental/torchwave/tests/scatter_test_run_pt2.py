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

import argparse
import os

import torch
from velox.experimental.torchwave.tests.scatter_test import ScatterTestPreproc


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write scatter_test.pt2 and scatter_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # out0: dim 0, base0[2:10:2, :] = src0 (4 rows, step 2). 8 < 12, fits.
    rows0, cols0 = 12, 5
    len0 = 4
    base0 = torch.arange(rows0 * cols0, dtype=torch.float32).reshape(rows0, cols0)
    src0 = (torch.arange(len0 * cols0, dtype=torch.float32) + 1000.0).reshape(
        len0, cols0
    )
    start0 = torch.tensor(2, dtype=torch.int64)

    # out1: dim 1, base1[:, 1:13:3] = src1 (4 cols, step 3). 10 < 14, fits.
    rows1, cols1 = 6, 14
    len1 = 4
    base1 = torch.arange(rows1 * cols1, dtype=torch.float32).reshape(rows1, cols1)
    src1 = (torch.arange(rows1 * len1, dtype=torch.float32) + 2000.0).reshape(
        rows1, len1
    )
    start1 = torch.tensor(1, dtype=torch.int64)

    inputs = (base0, src0, start0, base1, src1, start1)

    # Eager reference (slice_scatter is functional; base tensors are unchanged).
    module = ScatterTestPreproc()
    results = [o.clone() for o in module(*inputs)]
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={tuple(r.shape)}, dtype={r.dtype}")

    results_path = os.path.join(output_dir, "scatter_test_results.pt")
    torch.save(results, results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "scatter_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
