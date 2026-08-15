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
from velox.experimental.torchwave.tests.mixed_type_minmax_test import (
    MixedTypeMinMaxTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write mixed_type_minmax_test.pt2 and mixed_type_minmax_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducibility.
    torch.manual_seed(0)

    # Generate test inputs.
    a = torch.randint(0, 100, (64,), dtype=torch.int32)
    b = torch.randint(0, 100, (64,), dtype=torch.int64)

    # Run in eager mode.
    module = MixedTypeMinMaxTestPreproc()
    min_out, max_out = module(a, b)
    print(f"Eager results: min={min_out[:10]}, max={max_out[:10]}")

    # Save eager results.
    results_path = os.path.join(output_dir, "mixed_type_minmax_test_results.pt")
    torch.save([min_out, max_out], results_path)
    print(f"Saved results to {results_path}")

    # Export via torch.export.
    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            (a, b),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    # Save as .pt2.
    pt2_path = os.path.join(output_dir, "mixed_type_minmax_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
