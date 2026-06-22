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
from velox.experimental.torchwave.tests.element_test import (
    ElementTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write element_test.pt2 and element_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Generate test inputs: sequential 1, 2, 3, ...
    a = torch.arange(1, 10001, dtype=torch.long)
    b = torch.arange(1, 10001, dtype=torch.long)
    c = torch.arange(1, 10001, dtype=torch.long)
    d = torch.arange(1, 100001, dtype=torch.long)
    e = torch.arange(1, 100001, dtype=torch.long)
    f = torch.arange(1, 100001, dtype=torch.long)

    # Run in eager mode
    module = ElementTestPreproc()
    o1, o2, o3, o4, o5 = module(a, b, c, d, e, f)
    print(
        f"Eager results: o1={o1[:5]}, o2={o2[:5]}, o3={o3[:5]}, o4={o4[:5]}, o5={o5[:5]}"
    )

    # Save eager results as a list of tensors (easy to load in C++).
    results_path = os.path.join(output_dir, "element_test_results.pt")
    torch.save([o1, o2, o3, o4, o5], results_path)
    print(f"Saved results to {results_path}")

    # Export via torch.export
    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            (a, b, c, d, e, f),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    # Save as open source PyTorch .pt2 file with sample inputs embedded.
    # The example inputs passed to torch.export.export() are automatically
    # included in the .pt2 archive.
    pt2_path = os.path.join(output_dir, "element_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
