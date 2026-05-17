# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import argparse
import os

import torch
from torch import nn, Tensor


class SelectBenchmark(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, a: Tensor, b: Tensor) -> list[Tensor]:
        results = []
        for _ in range(self.n):
            r = torch.ops.aten.masked_select(
                a + b,
                torch.ops.aten.lt.Scalar(
                    torch.ops.aten.remainder.Scalar(a, 10), 9
                ),
            )
            results.append(r)
        return results


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write select benchmark .pt2 and _results.pt files",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ns = [1, 2, 4, 8, 16, 32, 64, 128]
    sizes = [
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        128000,
        256000,
        512000,
        1024000,
    ]

    for n in ns:
        for size in sizes:
            name = f"select_{n}_{size}"
            print(f"Generating {name}...")

            a = torch.arange(0, size, dtype=torch.long)
            b = torch.arange(0, size, dtype=torch.long)

            module = SelectBenchmark(n)
            results = module(a, b)

            results_path = os.path.join(output_dir, f"{name}_results.pt")
            torch.save(results, results_path)

            with torch.no_grad():
                exported_program = torch.export.export(
                    module,
                    (a, b),
                    strict=False,
                )

            pt2_path = os.path.join(output_dir, f"{name}.pt2")
            torch.export.save(exported_program, pt2_path)
            print(
                f"  {name}: {len(exported_program.graph.nodes)} nodes, "
                f"{os.path.getsize(pt2_path)} bytes"
            )


if __name__ == "__main__":
    main()
