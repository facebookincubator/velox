# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import os

import torch
from velox.experimental.torchwave.tests.isin_test import IsinTest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write isin_test.pt2 and isin_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # int32, 1k elements, ~50% match rate
    elements_i32_1k = torch.arange(1, 1001, dtype=torch.int32)
    set_i32_1k = torch.arange(1, 2001, 2, dtype=torch.int32)

    # int64, 1k elements
    elements_i64_1k = torch.arange(1, 1001, dtype=torch.int64)
    set_i64_1k = torch.arange(500, 1500, dtype=torch.int64)

    # int32, 10k elements
    elements_i32_10k = torch.arange(1, 10001, dtype=torch.int32)
    set_i32_10k = torch.arange(1, 20001, 2, dtype=torch.int32)

    # int64, 100k elements
    elements_i64_100k = torch.arange(1, 100001, dtype=torch.int64)
    set_i64_100k = torch.arange(50000, 150000, dtype=torch.int64)

    # int32 with zero in both elements and set
    elements_i32_nz = torch.arange(0, 1000, dtype=torch.int32)
    set_i32_with_zero = torch.cat(
        [torch.tensor([0], dtype=torch.int32),
         torch.arange(1, 1001, 2, dtype=torch.int32)]
    )

    # int64 with zero in both elements and set
    elements_i64_nz = torch.arange(0, 1000, dtype=torch.int64)
    set_i64_with_zero = torch.cat(
        [torch.tensor([0], dtype=torch.int64),
         torch.arange(500, 1500, dtype=torch.int64)]
    )

    inputs = (
        elements_i32_1k, set_i32_1k,
        elements_i64_1k, set_i64_1k,
        elements_i32_10k, set_i32_10k,
        elements_i64_100k, set_i64_100k,
        elements_i32_nz, set_i32_with_zero,
        elements_i64_nz, set_i64_with_zero,
    )

    module = IsinTest()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        matches = r.sum().item()
        print(f"  [{i}] shape={r.shape}, dtype={r.dtype}, matches={matches}/{r.numel()}")

    results_path = os.path.join(output_dir, "isin_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            inputs,
            strict=False,
        )
    print(
        f"Export successful, graph has {len(exported_program.graph.nodes)} nodes"
    )

    pt2_path = os.path.join(output_dir, "isin_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
