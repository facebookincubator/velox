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
import random

import torch
from velox.experimental.torchwave.tests.element_shape_test import (
    ElementShapeTest,
)


def make_inputs(
    contiguous: bool,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    full = torch.randn(10, 9, 8, generator=gen)
    bc_dim0 = torch.randn(1, 9, 8, generator=gen)
    bc_dim1 = torch.randn(10, 1, 8, generator=gen)
    bc_dim2 = torch.randn(10, 9, 1, generator=gen)
    bc_all = torch.randn(1, 1, 1, generator=gen)
    scalar = torch.randn((), generator=gen)
    if not contiguous:
        rng = random.Random(seed)

        def make_noncontiguous(t: torch.Tensor) -> torch.Tensor:
            if t.dim() < 2:
                return t
            # Pick two dims with size > 1 to transpose through.
            # Transpose, make contiguous copy, transpose back.
            # Result has original shape but non-contiguous strides.
            eligible = [d for d in range(t.dim()) if t.size(d) > 1]
            if len(eligible) < 2:
                return t
            i, j = rng.sample(eligible, 2)
            return t.transpose(i, j).contiguous().transpose(i, j)

        full = make_noncontiguous(full)
        bc_dim0 = make_noncontiguous(bc_dim0)
        bc_dim1 = make_noncontiguous(bc_dim1)
        bc_dim2 = make_noncontiguous(bc_dim2)
        bc_all = make_noncontiguous(bc_all)
    return full, bc_dim0, bc_dim1, bc_dim2, bc_all, scalar


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument("--output_dir", default=default_dir)
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    module = ElementShapeTest()

    # Contiguous case: export and save results.
    full, bc_dim0, bc_dim1, bc_dim2, bc_all, scalar = make_inputs(
        contiguous=True, seed=42
    )
    results_contig = list(module(full, bc_dim0, bc_dim1, bc_dim2, bc_all, scalar))
    print(f"Contiguous results: {[r.shape for r in results_contig]}")

    torch.save(
        results_contig, os.path.join(output_dir, "element_shape_test_results.pt")
    )

    with torch.no_grad():
        exported = torch.export.export(
            module,
            (full, bc_dim0, bc_dim1, bc_dim2, bc_all, scalar),
            strict=False,
        )
    pt2_path = os.path.join(output_dir, "element_shape_test.pt2")
    torch.export.save(exported, pt2_path)
    print(f"Saved contiguous .pt2 to {pt2_path}")

    # Non-contiguous case: transpose random dims and save results.
    full_nc, bc0_nc, bc1_nc, bc2_nc, bca_nc, sc_nc = make_inputs(
        contiguous=False, seed=42
    )
    results_nc = list(module(full_nc, bc0_nc, bc1_nc, bc2_nc, bca_nc, sc_nc))
    print(f"Non-contiguous results: {[r.shape for r in results_nc]}")

    torch.save(results_nc, os.path.join(output_dir, "element_shape_nc_test_results.pt"))

    with torch.no_grad():
        exported_nc = torch.export.export(
            module,
            (full_nc, bc0_nc, bc1_nc, bc2_nc, bca_nc, sc_nc),
            strict=False,
        )
    nc_path = os.path.join(output_dir, "element_shape_nc_test.pt2")
    torch.export.save(exported_nc, nc_path)
    print(f"Saved non-contiguous .pt2 to {nc_path}")


if __name__ == "__main__":
    main()
