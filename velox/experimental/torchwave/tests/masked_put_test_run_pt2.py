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
from velox.experimental.torchwave.tests.masked_put_test import MaskedPutTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write masked_put_test.pt2 and masked_put_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    n = 10000
    source_f = (torch.arange(n, dtype=torch.float) % 20).float()
    source_l = torch.arange(n, dtype=torch.long) % 59
    values_f = (torch.arange(n, dtype=torch.float) % 7 + 100).float()
    values_l = torch.arange(n, dtype=torch.long) % 11 + 200
    flags1 = torch.arange(n) % 5 == 0
    flags2 = torch.arange(n) % 5 == 3
    input_f = (torch.arange(n, dtype=torch.float) % 13 + 50).float()
    scalar_val = torch.tensor([42.0])
    scalar_mask = torch.tensor([True])

    # 2D [90, 130] tensor with int32/long indices, 500 scatter positions.
    dest2d = torch.zeros(90, 130, dtype=torch.float)
    num_scatter = 500
    idx2d_0 = torch.arange(num_scatter, dtype=torch.int32) % 90
    idx2d_1 = torch.arange(num_scatter, dtype=torch.long) * 7 % 130
    vals2d = (torch.arange(num_scatter, dtype=torch.float) + 1000).float()

    inputs = (
        source_f,
        source_l,
        values_f,
        values_l,
        flags1,
        flags2,
        input_f,
        scalar_val,
        scalar_mask,
        dest2d,
        idx2d_0,
        idx2d_1,
        vals2d,
    )

    module = MaskedPutTest()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={r.shape}, dtype={r.dtype}")
        if r.dim() == 1:
            print(f"       first 10: {r[:10].tolist()}")
        else:
            print(f"       [0,:5]: {r[0, :5].tolist()}")
            print(f"       nonzero: {r.nonzero().shape[0]}")

    results_path = os.path.join(output_dir, "masked_put_test_results.pt")
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

    pt2_path = os.path.join(output_dir, "masked_put_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
