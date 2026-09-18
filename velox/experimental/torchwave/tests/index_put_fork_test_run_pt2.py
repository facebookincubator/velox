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
from velox.experimental.torchwave.tests.index_put_fork_test import IndexPutForkTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write index_put_fork_test.pt2 and _results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    n = 128
    a = torch.arange(n, dtype=torch.float)
    b = (torch.arange(n, dtype=torch.float) % 7).float()
    idx1 = torch.tensor([0, 5, 10, 20, 40], dtype=torch.long)
    vals1 = torch.arange(idx1.numel(), dtype=torch.float) + 100
    idx2 = torch.tensor([1, 6, 11, 21], dtype=torch.long)
    vals2 = torch.arange(idx2.numel(), dtype=torch.float) + 200
    idx3 = torch.tensor([2, 7, 12, 22, 42], dtype=torch.long)
    vals3 = torch.arange(idx3.numel(), dtype=torch.float) + 300
    idx4 = torch.tensor([3, 8, 13], dtype=torch.long)
    vals4 = torch.arange(idx4.numel(), dtype=torch.float) + 400

    inputs = (a, b, idx1, vals1, idx2, vals2, idx3, vals3, idx4, vals4)

    module = IndexPutForkTest()
    results = module(*tuple(t.clone() for t in inputs))
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={r.shape}, dtype={r.dtype}, first 10: {r[:10].tolist()}")

    results_path = os.path.join(output_dir, "index_put_fork_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            tuple(t.clone() for t in inputs),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "index_put_fork_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
