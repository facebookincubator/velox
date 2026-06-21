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
from velox.experimental.torchwave.tests.tensor_test import TensorTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write tensor_test.pt2 and tensor_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    m = 256
    k = 50
    a = (torch.arange(m, dtype=torch.float) % 13).float()
    x = torch.arange(k, dtype=torch.long)
    inputs = (a, x)

    module = TensorTest()
    results = module(*tuple(t.clone() for t in inputs))
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(
            f"  [{i}] shape={r.shape}, dtype={r.dtype}, first 5: {r.flatten()[:5].tolist()}"
        )

    results_path = os.path.join(output_dir, "tensor_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    # x's dim 0 is dynamic so sym_size(x, 0) stays symbolic and torch.tensor(n)
    # lowers to scalar_tensor instead of being folded to a constant.
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

    pt2_path = os.path.join(output_dir, "tensor_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
