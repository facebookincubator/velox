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
from velox.experimental.torchwave.tests.masked_select_empty_test import (
    MaskedSelectEmptyTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument("--output_dir", default=default_dir)
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # data[i] = i, flags[i] = (i % 3 == 0); end = 0 makes the sliced flags/data
    # dynamically empty at runtime.
    data = torch.arange(0, 10000, dtype=torch.long)
    flags = torch.arange(0, 10000) % 3 == 0
    end = torch.tensor(0, dtype=torch.long)

    module = MaskedSelectEmptyTestPreproc()
    result = module(data, flags, end)
    print(f"Eager result: shape={tuple(result.shape)}, numel={result.numel()}")

    results_path = os.path.join(output_dir, "masked_select_empty_test_results.pt")
    torch.save([result], results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            (data, flags, end),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")
    # Show the graph so the dynamic slice feeding masked_select is visible.
    print(exported_program.graph)

    pt2_path = os.path.join(output_dir, "masked_select_empty_test.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"Saved .pt2 to {pt2_path} ({os.path.getsize(pt2_path)} bytes)")


if __name__ == "__main__":
    main()
