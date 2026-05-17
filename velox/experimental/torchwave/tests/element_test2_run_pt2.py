# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import os

import torch
from velox.experimental.torchwave.tests.element_test2 import (
    ElementTest2Preproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write element_test2.pt2 and element_test2_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Generate test inputs.
    a = torch.arange(1, 10001, dtype=torch.long)
    b = torch.arange(1, 10001, dtype=torch.long)
    c = torch.arange(1, 10001, dtype=torch.float32)
    d = torch.arange(1, 10001, dtype=torch.float32)

    inputs = (a, b, c, d)

    # Run in eager mode.
    module = ElementTest2Preproc()
    results = module(*inputs)
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={r.shape}, dtype={r.dtype}, value={r[:5]}")

    # Save eager results.
    results_path = os.path.join(output_dir, "element_test2_results.pt")
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
    pt2_path = os.path.join(output_dir, "element_test2.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")


if __name__ == "__main__":
    main()
