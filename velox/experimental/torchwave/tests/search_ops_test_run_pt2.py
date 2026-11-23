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
from velox.experimental.torchwave.tests.search_ops_test import SearchOpsTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write search_ops_test.pt2 and search_ops_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Float bucketize: boundaries are 1-D ascending; queries include values that
    # equal a boundary (1.0, 3.0, 5.0), out-of-range values (0.5, 10.0), and a
    # NaN (which lands past the last bucket in eager).
    bounds_f = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], dtype=torch.float)
    query_f = torch.tensor(
        [0.5, 1.0, 2.0, 3.0, 5.0, 9.0, 10.0, float("nan")], dtype=torch.float
    )

    # Int bucketize with int32 output.
    bounds_i = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    query_i = torch.tensor([5, 10, 15, 20, 45], dtype=torch.int64)

    # 1-D searchsorted.
    sorted_1d = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float)
    query_1d = torch.tensor([1.0, 2.0, 5.0, 8.0, 9.0], dtype=torch.float)

    # 2-D (multi-row) searchsorted: each query row searches the matching sorted
    # row.
    sorted_2d = torch.tensor([[1.0, 3.0, 5.0], [10.0, 20.0, 30.0]], dtype=torch.float)
    query_2d = torch.tensor([[0.0, 3.0, 6.0], [15.0, 30.0, 5.0]], dtype=torch.float)

    inputs = (
        query_f,
        bounds_f,
        query_i,
        bounds_i,
        query_1d,
        sorted_1d,
        query_2d,
        sorted_2d,
    )

    module = SearchOpsTest()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={tuple(r.shape)}, dtype={r.dtype}")

    results_path = os.path.join(output_dir, "search_ops_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "search_ops_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
