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
from velox.experimental.torchwave.tests.dedup_test import DedupTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument("--output_dir", default=default_dir)
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    x1 = torch.randn(5, 10)
    x2 = torch.randn(7, 10)
    x3 = torch.randn(5, 6)
    x4 = torch.randn(7, 6)
    inputs = (x1, x2, x3, x4)

    module = DedupTest()
    with torch.no_grad():
        results = module(*inputs)

    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={list(r.shape)}, dtype={r.dtype}")

    results_path = os.path.join(output_dir, "dedup_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    inputs_path = os.path.join(output_dir, "dedup_test_inputs.pt")
    torch.save(list(inputs), inputs_path)
    print(f"Saved inputs to {inputs_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "dedup_test.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"Saved .pt2 to {pt2_path} ({os.path.getsize(pt2_path)} bytes)")


if __name__ == "__main__":
    main()
