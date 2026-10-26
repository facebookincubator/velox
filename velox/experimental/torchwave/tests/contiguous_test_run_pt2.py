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
from velox.experimental.torchwave.tests.contiguous_test import ContiguousTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write contiguous_test.pt2 and contiguous_test_results.pt",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(0)
    inputs = (
        torch.randn(70, 45),
        torch.randn(300, 257),
        torch.randint(-1000, 1000, (40, 33), dtype=torch.int64),
        torch.randn(12, 10),
        torch.randn(6, 5, 4),
    )

    module = ContiguousTest()
    results = module(*inputs)
    results_path = os.path.join(args.output_dir, "contiguous_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(exported_program.graph)
    pt2_path = os.path.join(args.output_dir, "contiguous_test.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"Saved {pt2_path}")


if __name__ == "__main__":
    main()
