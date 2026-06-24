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
from velox.experimental.torchwave.tests.element_test_100 import (
    ElementTest100,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write element_test_100.pt2 and element_test_100_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Generate test inputs: s0-s49 (1000), m0-m29 (10000), l0-l19 (100000)
    inputs = []
    for _i in range(50):
        inputs.append(torch.arange(0, 1000, dtype=torch.long))
    for _i in range(30):
        inputs.append(torch.arange(0, 10000, dtype=torch.long))
    for _i in range(20):
        inputs.append(torch.arange(0, 100000, dtype=torch.long))

    # Run in eager mode
    module = ElementTest100()
    results = module(*inputs)
    print(f"Eager mode produced {len(results)} outputs")
    for i, r in enumerate(results):
        print(f"  o{i}: shape={r.shape}, first5={r[:5].tolist()}")

    # Save eager results
    results_path = os.path.join(output_dir, "element_test_100_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    # Export via torch.export
    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            tuple(inputs),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    # Save as .pt2
    pt2_path = os.path.join(output_dir, "element_test_100.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
