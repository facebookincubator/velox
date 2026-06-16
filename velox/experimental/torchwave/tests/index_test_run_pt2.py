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
from velox.experimental.torchwave.tests.index_test import IndexTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write index_test.pt2 and index_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1D: 20000-element tensor, scatter 10000 values at every 2nd position.
    dest1d = torch.zeros(20000, dtype=torch.long)
    idx1d_0 = torch.arange(0, 20000, 2, dtype=torch.long)
    vals1d = torch.arange(1, 10001, dtype=torch.long)

    # 2D: 200x100 tensor (20000 elements), scatter 5000 values at unique
    # positions using every 4th flat index. dim0=int32, dim1=long.
    dest2d = torch.zeros(200, 100, dtype=torch.long)
    idx2d_flat = torch.arange(0, 20000, 4)
    idx2d_0 = (idx2d_flat // 100).to(torch.int32)
    idx2d_1 = (idx2d_flat % 100).to(torch.long)
    vals2d = torch.arange(10001, 15001, dtype=torch.long)

    # 3D: 20x50x20 tensor (20000 elements), scatter 2000 values at unique
    # positions using every 10th flat index. dim0=int32, dim1=long, dim2=int32.
    dest3d = torch.zeros(20, 50, 20, dtype=torch.long)
    idx3d_flat = torch.arange(0, 20000, 10)
    idx3d_0 = (idx3d_flat // (50 * 20)).to(torch.int32)
    idx3d_1 = ((idx3d_flat % (50 * 20)) // 20).to(torch.long)
    idx3d_2 = (idx3d_flat % 20).to(torch.int32)
    vals3d = torch.arange(20001, 22001, dtype=torch.long)

    # Broadcast case: 20x50x20 dest, 500 scatter positions.
    # idx_bc_0: 1-element tensor (broadcasts to all positions) = always row 5.
    # idx_bc_1: 500 elements with stride 2 (from a 1000-element backing tensor).
    # idx_bc_2: 500 elements, contiguous.
    # vals_bc: 1-element tensor (broadcasts, fills all positions with same value).
    dest_bc = torch.zeros(20, 50, 20, dtype=torch.long)
    idx_bc_0 = torch.tensor([5], dtype=torch.int32)
    backing = torch.arange(0, 1000, dtype=torch.long)
    idx_bc_1 = backing[::2][:500] % 50
    idx_bc_2 = torch.arange(500, dtype=torch.int32) % 20
    vals_bc = torch.tensor([999], dtype=torch.long)

    inputs = (
        dest1d,
        idx1d_0,
        vals1d,
        dest2d,
        idx2d_0,
        idx2d_1,
        vals2d,
        dest3d,
        idx3d_0,
        idx3d_1,
        idx3d_2,
        vals3d,
        dest_bc,
        idx_bc_0,
        idx_bc_1,
        idx_bc_2,
        vals_bc,
    )

    module = IndexTest()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(
            f"  [{i}] shape={r.shape}, dtype={r.dtype}, nonzero={r.nonzero().shape[0]}"
        )

    results_path = os.path.join(output_dir, "index_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            inputs,
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "index_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
