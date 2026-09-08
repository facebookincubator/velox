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
from velox.experimental.torchwave.tests.cumsum_select3d_test import (
    CumsumSelect3dTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write cumsum_select3d_test.pt2 and cumsum_select3d_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 3D so x[:, 0, 1] is a doubly-strided view; 1000 rows spans multiple
    # blocks.  Element [i, j, k] = 12*i + 3*j + k, so the selected column has
    # distinctive values a contiguous (stride-ignoring) misread would not match.
    x = torch.arange(1000 * 4 * 3, dtype=torch.int64).reshape(1000, 4, 3)

    # Run in eager mode.
    module = CumsumSelect3dTestPreproc()
    result = module(x)
    print(
        f"Eager result: shape={result.shape}, dtype={result.dtype}, "
        f"first5={result[:5].tolist()}, last={result[-1].item()}"
    )

    # Save eager results.
    results_path = os.path.join(output_dir, "cumsum_select3d_test_results.pt")
    torch.save([result], results_path)
    print(f"Saved results to {results_path}")

    # Export via torch.export.
    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            (x,),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    # Save as .pt2.
    pt2_path = os.path.join(output_dir, "cumsum_select3d_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
