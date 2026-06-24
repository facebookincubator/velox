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
from velox.experimental.torchwave.tests.in_place_test import (
    InPlaceTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write in_place_test.pt2 and in_place_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    n = 1000
    a = torch.arange(1, n + 1, dtype=torch.float32)
    b = torch.arange(1, n + 1, dtype=torch.float32)
    y = torch.arange(1, n + 1, dtype=torch.float32)
    yy = torch.arange(1, n + 1, dtype=torch.float32)
    ww = torch.arange(1, n + 1, dtype=torch.float32)
    inputs = (a, b, y, yy, ww)

    # Eager reference. The module mutates a/y/yy in place, so run it on copies
    # and snapshot every returned tensor (several outputs alias mutated buffers).
    module = InPlaceTestPreproc()
    outs = module(*(t.clone() for t in inputs))
    results = [o.clone() for o in outs]
    print(f"Eager results (first elem each): {[float(r[0]) for r in results]}")

    results_path = os.path.join(output_dir, "in_place_test_results.pt")
    torch.save(results, results_path)
    print(f"Saved results to {results_path}")

    # Export. NOTE: torch.export functionalizes by default, which would rewrite
    # the in-place ops (add_) into functional ops + input-mutation copy_, and
    # this test is specifically meant to exercise the non-functionalized,
    # in-place path that torchwave must accept. If the exported graph does not
    # retain the aten add_/view aliasing chain, switch the capture here to the
    # non-functionalizing path used for the roo_train_preproc graphs.
    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            tuple(t.clone() for t in inputs),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "in_place_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
