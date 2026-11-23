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
from velox.experimental.torchwave.tests.index_put_chain_test import IndexPutChainTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write index_put_chain_test.pt2 and _results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    n = 128
    base = torch.zeros(n, dtype=torch.float)
    idx2 = torch.tensor([0, 5, 10, 20, 40, 80, 127], dtype=torch.long)
    vals2 = torch.arange(idx2.numel(), dtype=torch.float) + 200
    idx3 = torch.tensor([1, 6, 11, 21, 41, 81], dtype=torch.long)
    vals3 = torch.arange(idx3.numel(), dtype=torch.float) + 300
    idx4 = torch.tensor([2, 7, 12, 22, 42], dtype=torch.long)
    vals4 = torch.arange(idx4.numel(), dtype=torch.float) + 400

    inputs = (base, idx2, vals2, idx3, vals3, idx4, vals4)

    module = IndexPutChainTest()
    result = module(*tuple(t.clone() for t in inputs))
    print(f"Eager result: shape={result.shape}, dtype={result.dtype}")
    print(f"  first 10: {result[:10].tolist()}")

    results_path = os.path.join(output_dir, "index_put_chain_test_results.pt")
    torch.save([result], results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            tuple(t.clone() for t in inputs),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "index_put_chain_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
