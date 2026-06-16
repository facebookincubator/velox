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
from velox.experimental.torchwave.tests.numel_test import NumelTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write numel_test.pt2 and numel_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    m = 256
    # x is 2-D so numel = d0 * d1 is a genuine numel (not just sym_size(0)).
    k0 = 12
    k1 = 5
    a = (torch.arange(m, dtype=torch.float) % 13).float()
    x = torch.arange(k0 * k1, dtype=torch.long).reshape(k0, k1)
    inputs = (a, x)

    module = NumelTest()
    results = module(*tuple(t.clone() for t in inputs))
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        if isinstance(r, torch.Tensor):
            print(
                f"  [{i}] tensor shape={r.shape}, dtype={r.dtype}, first 5: {r.flatten()[:5].tolist()}"
            )
        else:
            print(f"  [{i}] scalar {type(r).__name__} = {r}")

    results_path = os.path.join(output_dir, "numel_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    # x's dim 0 is dynamic so numel(x) stays symbolic.
    dynamic_shapes = {"a": None, "x": {0: Dim("xn")}}
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

    pt2_path = os.path.join(output_dir, "numel_test.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"Saved .pt2 to {pt2_path} ({os.path.getsize(pt2_path)} bytes)")


if __name__ == "__main__":
    main()
