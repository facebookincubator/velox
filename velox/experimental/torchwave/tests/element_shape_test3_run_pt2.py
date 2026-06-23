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
from velox.experimental.torchwave.tests.element_shape_test3 import (
    ElementShapeTest3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument("--output_dir", default=default_dir)
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize with arange, then reshape to the final broadcast shapes.
    t1 = torch.arange(100, dtype=torch.float).reshape(100)
    t2 = torch.arange(20, dtype=torch.float).reshape(20, 1)
    t3 = torch.arange(10, dtype=torch.float).reshape(10, 1, 1)

    module = ElementShapeTest3()
    result = module(t1, t2, t3)
    print(f"result shape={tuple(result.shape)} dtype={result.dtype}")

    results_path = os.path.join(output_dir, "element_shape_test3_results.pt")
    torch.save([result], results_path)
    print(f"Saved results to {results_path}")

    with torch.no_grad():
        exported = torch.export.export(module, (t1, t2, t3), strict=False)
    print(f"Export successful, graph has {len(exported.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "element_shape_test3.pt2")
    torch.export.save(exported, pt2_path)
    print(f"Saved .pt2 to {pt2_path}")


if __name__ == "__main__":
    main()
