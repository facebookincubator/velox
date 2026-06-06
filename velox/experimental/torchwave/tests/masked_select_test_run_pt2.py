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
from velox.experimental.torchwave.tests.masked_select_test import (
    MaskedSelectTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write masked_select_test.pt2 and masked_select_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Generate test inputs: a[i] = i % 200, b[i] = i % 100
    a = torch.arange(0, 10000, dtype=torch.long) % 200
    b = torch.arange(0, 10000, dtype=torch.long) % 100

    # Run in eager mode
    module = MaskedSelectTestPreproc()
    result = module(a, b)
    print(f"Eager result: shape={result.shape}, first 10={result[:10]}")

    # Save eager results as a list of tensors (easy to load in C++).
    results_path = os.path.join(output_dir, "masked_select_test_results.pt")
    torch.save([result], results_path)
    print(f"Saved results to {results_path}")

    # Export via torch.export
    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            (a, b),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    # Save as open source PyTorch .pt2 file with sample inputs embedded.
    pt2_path = os.path.join(output_dir, "masked_select_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
