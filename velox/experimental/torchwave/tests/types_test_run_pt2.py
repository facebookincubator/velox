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
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import os

import torch
from velox.experimental.torchwave.tests.types_test import (
    TypesTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write types_test.pt2 and types_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    n = 10000

    # Case 1: Long * Half — values > 2048 lose precision in half but not float.
    a1 = torch.zeros(n, dtype=torch.long)
    b1 = torch.arange(2049, 2049 + n, dtype=torch.long)
    c1 = torch.ones(n, dtype=torch.float16)

    # Case 2: Long * BFloat16 — values > 256 lose precision in bf16 but not float.
    a2 = torch.zeros(n, dtype=torch.long)
    b2 = torch.arange(257, 257 + n, dtype=torch.long)
    c2 = torch.ones(n, dtype=torch.bfloat16)

    # Case 3: Int * Half — same precision boundary as Case 1.
    a3 = torch.zeros(n, dtype=torch.int)
    b3 = torch.arange(2049, 2049 + n, dtype=torch.int)
    c3 = torch.ones(n, dtype=torch.float16)

    # Case 4: Long * Float — baseline, both agree on Float.
    a4 = torch.zeros(n, dtype=torch.long)
    b4 = torch.arange(16777217, 16777217 + n, dtype=torch.long)
    c4 = torch.ones(n, dtype=torch.float32)

    module = TypesTestPreproc()
    o1, o2, o3, o4 = module(a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4)
    print(f"o1 (Long*Half->Float):    dtype={o1.dtype}, first={o1[:3].tolist()}")
    print(f"o2 (Long*BF16->Float):    dtype={o2.dtype}, first={o2[:3].tolist()}")
    print(f"o3 (Int*Half->Float):     dtype={o3.dtype}, first={o3[:3].tolist()}")
    print(f"o4 (Long*Float baseline): dtype={o4.dtype}, first={o4[:3].tolist()}")

    results_path = os.path.join(output_dir, "types_test_results.pt")
    torch.save([o1, o2, o3, o4], results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            (a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "types_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
