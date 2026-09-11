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
from velox.experimental.torchwave.tests.cat_dedup_test import CatDedupTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument("--output_dir", default=default_dir)
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(1)
    # Uneven repeat counts, so the two concats' operands differ in length and a
    # destination bound to the wrong band shows up as a shifted result rather
    # than as equal-sized garbage that happens to line up.
    reps = torch.arange(64, dtype=torch.long) % 3 + 1
    inputs = (
        torch.arange(0, 64, dtype=torch.long),
        torch.arange(100, 164, dtype=torch.long),
        reps,
        # Distinct values, so a region left unwritten cannot pass for a region
        # written from the wrong source.
        torch.arange(9000, 9016, dtype=torch.long),
        torch.arange(7000, 7024, dtype=torch.long),
        torch.arange(8000, 8024, dtype=torch.long),
    )

    module = CatDedupTest()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={tuple(r.shape)}, dtype={r.dtype}")

    results_path = os.path.join(output_dir, "cat_dedup_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "cat_dedup_test.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"Saved .pt2 to {pt2_path} ({os.path.getsize(pt2_path)} bytes)")


if __name__ == "__main__":
    main()
