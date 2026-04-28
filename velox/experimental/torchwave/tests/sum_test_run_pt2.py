# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import os

import torch
from velox.experimental.torchwave.tests.sum_test import SumTestPreproc


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write sum_test.pt2 and sum_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Generate test inputs.
    float_1k = torch.arange(1, 1001, dtype=torch.float32)
    float_10k = torch.arange(1, 10001, dtype=torch.float32)
    float_100k = torch.arange(1, 100001, dtype=torch.float32)
    int_1k = torch.arange(1, 1001, dtype=torch.long)
    int_10k = torch.arange(1, 10001, dtype=torch.long)
    int_100k = torch.arange(1, 100001, dtype=torch.long)

    inputs = (float_1k, float_10k, float_100k, int_1k, int_10k, int_100k)

    # Run in eager mode.
    module = SumTestPreproc()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        if isinstance(r, torch.Tensor):
            print(f"  [{i}] shape={r.shape}, dtype={r.dtype}, value={r}")
        else:
            print(f"  [{i}] scalar, type={type(r).__name__}, value={r}")

    # Save eager results.
    results_path = os.path.join(output_dir, "sum_test_results.pt")
    torch.save(list(results), results_path)
    print(f"Saved results to {results_path}")

    # Export via torch.export.
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

    # Save as .pt2.
    pt2_path = os.path.join(output_dir, "sum_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
