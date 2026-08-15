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
from torch.export import Dim
from velox.experimental.torchwave.tests.view_interleave_test import ViewInterleaveTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write view_interleave_test.pt2 and "
        "view_interleave_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    r = 24
    c = 40
    # A few NaNs/infs so nan_to_num actually does work.
    x = (torch.arange(r * c, dtype=torch.float) % 97).reshape(r, c) - 40.0
    x[0, 0] = float("nan")
    x[1, 3] = float("inf")
    x[2, 5] = float("-inf")
    inputs = (x,)

    module = ViewInterleaveTest()
    results = module(*tuple(t.clone() for t in inputs))
    print(f"Eager results ({len(results)} outputs):")
    for i, res in enumerate(results):
        print(
            f"  [{i}] shape={res.shape}, dtype={res.dtype}, "
            f"first 5: {res.flatten()[:5].tolist()}"
        )

    results_path = os.path.join(output_dir, "view_interleave_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    # x's dim 0 is dynamic so the row count stays symbolic.
    dynamic_shapes = {"x": {0: Dim("r")}}
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            tuple(t.clone() for t in inputs),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")
    for node in exported_program.graph.nodes:
        if node.op == "call_function":
            print(f"  node: {node.target}")

    pt2_path = os.path.join(output_dir, "view_interleave_test.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"Saved .pt2 to {pt2_path} ({os.path.getsize(pt2_path)} bytes)")


if __name__ == "__main__":
    main()
