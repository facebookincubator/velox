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
from velox.experimental.torchwave.tests.repeat_interleave_test import (
    RepeatInterleaveTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument("--output_dir", default=default_dir)
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data_long = torch.arange(1, 1001, dtype=torch.long)
    repeats_long = torch.arange(0, 1000, dtype=torch.long) % 4
    data_long2 = torch.arange(1, 1001, dtype=torch.long)
    repeats_int = torch.arange(0, 1000, dtype=torch.int32) % 4
    data_float = torch.randn(500, dtype=torch.float32)
    repeats_long2 = torch.arange(0, 500, dtype=torch.long) % 3
    data_small = torch.arange(1, 11, dtype=torch.long)
    repeats_small = torch.tensor([5, 0, 3, 1, 0, 7, 2, 4, 0, 6], dtype=torch.long)

    inputs = (
        data_long,
        repeats_long,
        data_long2,
        repeats_int,
        data_float,
        repeats_long2,
        data_small,
        repeats_small,
    )

    module = RepeatInterleaveTestPreproc()
    o1, o2, o3, o4 = module(*inputs)
    print("Eager results:")
    print(f"  o1 (long/long): shape={o1.shape}, first5={o1[:5].tolist()}")
    print(f"  o2 (long/int32): shape={o2.shape}, first5={o2[:5].tolist()}")
    print(f"  o3 (float/long): shape={o3.shape}, first5={o3[:5].tolist()}")
    print(f"  o4 (small): shape={o4.shape}, values={o4.tolist()}")

    results_path = os.path.join(output_dir, "repeat_interleave_test_results.pt")
    torch.save([o1, o2, o3, o4], results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "repeat_interleave_test.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"Saved .pt2 to {pt2_path} ({os.path.getsize(pt2_path)} bytes)")


if __name__ == "__main__":
    main()
