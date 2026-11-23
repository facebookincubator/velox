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
from velox.experimental.torchwave.tests.index_elt_default_test import (
    IndexEltDefaultTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument("--output_dir", default=default_dir)
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    var = torch.arange(0, 10, dtype=torch.long)
    var2 = torch.arange(10, 20, dtype=torch.long)
    # cat([var, pad]) has length 14 and cat([var, var2, pad]) length 24; every
    # index is < 14 so both eager cats are in range. 10 and 13 fall in the pad
    # region of the two-operand cat, exercising the default.
    idx = torch.tensor([0, 3, 9, 10, 13], dtype=torch.long)

    module = IndexEltDefaultTestPreproc()
    matched, unmatched = module(var, var2, idx)
    print(f"matched={matched.tolist()} unmatched={unmatched.tolist()}")

    results_path = os.path.join(output_dir, "index_elt_default_test_results.pt")
    torch.save([matched, unmatched], results_path)
    print(f"Saved results to {results_path}")

    with torch.no_grad():
        exported_program = torch.export.export(module, (var, var2, idx), strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "index_elt_default_test.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"Saved .pt2 to {pt2_path} ({os.path.getsize(pt2_path)} bytes)")


if __name__ == "__main__":
    main()
