# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import os

import torch
from velox.experimental.torchwave.tests.cat_test import (
    CatTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write cat_test.pt2 and cat_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    a = torch.arange(0, 10000, dtype=torch.long) % 200
    b = torch.arange(0, 10000, dtype=torch.long) % 100

    module = CatTestPreproc()
    o1, o2, o3 = module(a, b)
    print(f"Eager results: o1.shape={o1.shape}, o2.shape={o2.shape}, o3.shape={o3.shape}")
    print(f"  o1[:5]={o1[:5]}, o2[:5]={o2[:5]}, o3[:5]={o3[:5]}")

    results_path = os.path.join(output_dir, "cat_test_results.pt")
    torch.save([o1, o2, o3], results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            (a, b),
            strict=False,
        )
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "cat_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
