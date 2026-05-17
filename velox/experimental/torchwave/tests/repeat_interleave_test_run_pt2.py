# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import os

import torch
from velox.experimental.torchwave.tests.repeat_interleave_test import (
    RepeatInterleaveTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write repeat_interleave_test.pt2 and repeat_interleave_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data = torch.arange(1, 1001, dtype=torch.long)
    repeats = torch.arange(0, 1000, dtype=torch.long) % 4

    inputs = (data, repeats)

    module = RepeatInterleaveTestPreproc()
    result = module(*inputs)
    print(f"Eager result: shape={result.shape}, dtype={result.dtype}, first={result[:5]}")

    results_path = os.path.join(output_dir, "repeat_interleave_test_results.pt")
    torch.save([result], results_path)
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

    pt2_path = os.path.join(output_dir, "repeat_interleave_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
