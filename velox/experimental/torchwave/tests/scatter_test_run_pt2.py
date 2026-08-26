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
from velox.experimental.torchwave.tests.scatter_test import (
    ScatterAddCgConsumerTestPreproc,
    ScatterAddTestPreproc,
    ScatterSrcTestPreproc,
    ScatterTestPreproc,
    SelectScatterTestPreproc,
    SliceScatterDim1ViewTestPreproc,
    SliceScatterInPlaceTestPreproc,
    SliceScatterOpenEndTestPreproc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write scatter_test.pt2 and scatter_test_results.pt",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # out0: dim 0, base0[2:10:2, :] = src0 (4 rows, step 2). 8 < 12, fits.
    rows0, cols0 = 12, 5
    len0 = 4
    base0 = torch.arange(rows0 * cols0, dtype=torch.float32).reshape(rows0, cols0)
    src0 = (torch.arange(len0 * cols0, dtype=torch.float32) + 1000.0).reshape(
        len0, cols0
    )
    start0 = torch.tensor(2, dtype=torch.int64)

    # out1: dim 1, base1[:, 1:13:3] = src1 (4 cols, step 3). 10 < 14, fits.
    rows1, cols1 = 6, 14
    len1 = 4
    base1 = torch.arange(rows1 * cols1, dtype=torch.float32).reshape(rows1, cols1)
    src1 = (torch.arange(rows1 * len1, dtype=torch.float32) + 2000.0).reshape(
        rows1, len1
    )
    start1 = torch.tensor(1, dtype=torch.int64)

    inputs = (base0, src0, start0, base1, src1, start1)

    # Eager reference (slice_scatter is functional; base tensors are unchanged).
    module = ScatterTestPreproc()
    results = [o.clone() for o in module(*inputs)]
    print(f"Eager results ({len(results)} outputs):")
    for i, r in enumerate(results):
        print(f"  [{i}] shape={tuple(r.shape)}, dtype={r.dtype}")

    results_path = os.path.join(output_dir, "scatter_test_results.pt")
    torch.save(results, results_path)
    print(f"Saved results to {results_path}")

    print("Exporting via torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(module, inputs, strict=False)
    print(f"Export successful, graph has {len(exported_program.graph.nodes)} nodes")

    pt2_path = os.path.join(output_dir, "scatter_test.pt2")
    print(f"Saving exported program to {pt2_path}")
    torch.export.save(exported_program, pt2_path)
    print(f"Successfully saved .pt2 to {pt2_path}")
    print(f"File size: {os.path.getsize(pt2_path)} bytes")

    # select_scatter: out0 in-place (base0 = a0+a1 is a dead intermediate), out1
    # copy (base1 = b0+b1 is also returned, so it stays live).
    r0, c0 = 12, 5
    a0 = torch.arange(r0 * c0, dtype=torch.float32).reshape(r0, c0)
    a1 = torch.ones(r0 * c0, dtype=torch.float32).reshape(r0, c0)
    src0 = torch.arange(c0, dtype=torch.float32) + 1000.0
    r1, c1 = 6, 14
    b0 = torch.arange(r1 * c1, dtype=torch.float32).reshape(r1, c1)
    b1 = torch.ones(r1 * c1, dtype=torch.float32).reshape(r1, c1)
    src1 = torch.arange(r1, dtype=torch.float32) + 2000.0
    sel_inputs = (a0, a1, src0, b0, b1, src1)
    sel_module = SelectScatterTestPreproc()
    sel_results = [o.clone() for o in sel_module(*sel_inputs)]
    torch.save(
        sel_results,
        os.path.join(output_dir, "select_scatter_test_results.pt"),
    )
    with torch.no_grad():
        sel_ep = torch.export.export(sel_module, sel_inputs, strict=False)
    sel_pt2 = os.path.join(output_dir, "select_scatter_test.pt2")
    torch.export.save(sel_ep, sel_pt2)
    print(f"select_scatter: {len(sel_ep.graph.nodes)} nodes -> {sel_pt2}")

    # slice_scatter in-place: out0 in-place (base0 = a0+a1 is a dead intermediate,
    # slice dim 0 [2:10:2] -> 4 rows), out1 copy (base1 = b0+b1 is also returned,
    # slice dim 1 [1:13:3] -> 4 cols). Exercises the elided-clone / intra-op
    # materialized-self path that scatterTest's graph-input bases never hit.
    sr0, sc0 = 12, 5
    sa0 = torch.arange(sr0 * sc0, dtype=torch.float32).reshape(sr0, sc0)
    sa1 = torch.ones(sr0 * sc0, dtype=torch.float32).reshape(sr0, sc0)
    ssrc0 = (torch.arange(4 * sc0, dtype=torch.float32) + 1000.0).reshape(4, sc0)
    sr1, sc1 = 6, 14
    sb0 = torch.arange(sr1 * sc1, dtype=torch.float32).reshape(sr1, sc1)
    sb1 = torch.ones(sr1 * sc1, dtype=torch.float32).reshape(sr1, sc1)
    ssrc1 = (torch.arange(sr1 * 4, dtype=torch.float32) + 2000.0).reshape(sr1, 4)
    slice_inputs = (sa0, sa1, ssrc0, sb0, sb1, ssrc1)
    slice_module = SliceScatterInPlaceTestPreproc()
    slice_results = [o.clone() for o in slice_module(*slice_inputs)]
    torch.save(
        slice_results,
        os.path.join(output_dir, "slice_scatter_inplace_test_results.pt"),
    )
    with torch.no_grad():
        slice_ep = torch.export.export(slice_module, slice_inputs, strict=False)
    slice_pt2 = os.path.join(output_dir, "slice_scatter_inplace_test.pt2")
    torch.export.save(slice_ep, slice_pt2)
    print(f"slice_scatter in-place: {len(slice_ep.graph.nodes)} nodes -> {slice_pt2}")

    # scatter.src: a permutation index along 'dim' (each destination written
    # once) so the overwrite result is deterministic. dim 0 reverses rows, dim 1
    # reverses cols.
    xr0, xc0 = 12, 5
    xa0 = torch.arange(xr0 * xc0, dtype=torch.float32).reshape(xr0, xc0)
    xa1 = torch.ones(xr0 * xc0, dtype=torch.float32).reshape(xr0, xc0)
    xidx0 = (
        (xr0 - 1 - torch.arange(xr0, dtype=torch.int64))
        .reshape(xr0, 1)
        .expand(xr0, xc0)
        .contiguous()
    )
    xsrc0 = (torch.arange(xr0 * xc0, dtype=torch.float32) + 1000.0).reshape(xr0, xc0)
    xr1, xc1 = 6, 14
    xb0 = torch.arange(xr1 * xc1, dtype=torch.float32).reshape(xr1, xc1)
    xb1 = torch.ones(xr1 * xc1, dtype=torch.float32).reshape(xr1, xc1)
    xidx1 = (
        (xc1 - 1 - torch.arange(xc1, dtype=torch.int64))
        .reshape(1, xc1)
        .expand(xr1, xc1)
        .contiguous()
    )
    xsrc1 = (torch.arange(xr1 * xc1, dtype=torch.float32) + 2000.0).reshape(xr1, xc1)
    src_inputs = (xa0, xa1, xidx0, xsrc0, xb0, xb1, xidx1, xsrc1)
    src_module = ScatterSrcTestPreproc()
    src_results = [o.clone() for o in src_module(*src_inputs)]
    torch.save(src_results, os.path.join(output_dir, "scatter_src_test_results.pt"))
    with torch.no_grad():
        src_ep = torch.export.export(src_module, src_inputs, strict=False)
    src_pt2 = os.path.join(output_dir, "scatter_src_test.pt2")
    torch.export.save(src_ep, src_pt2)
    print(f"scatter.src: {len(src_ep.graph.nodes)} nodes -> {src_pt2}")

    # scatter_add: out0 clone elided (base0 = aa0+aa1 dead, src independent), out1
    # clone kept (src is base1 itself -> shares base1's storage). Duplicate
    # indices (i%2 / j%2) accumulate, exercising the atomic add.
    yr0, yc0 = 12, 5
    ya0 = torch.arange(yr0 * yc0, dtype=torch.float32).reshape(yr0, yc0)
    ya1 = torch.ones(yr0 * yc0, dtype=torch.float32).reshape(yr0, yc0)
    yidx0 = (
        (torch.arange(yr0, dtype=torch.int64) % 2)
        .reshape(yr0, 1)
        .expand(yr0, yc0)
        .contiguous()
    )
    ysrc0 = (torch.arange(yr0 * yc0, dtype=torch.float32) + 1000.0).reshape(yr0, yc0)
    yr1, yc1 = 6, 14
    yb0 = torch.arange(yr1 * yc1, dtype=torch.float32).reshape(yr1, yc1)
    yb1 = torch.ones(yr1 * yc1, dtype=torch.float32).reshape(yr1, yc1)
    yidx1 = (
        (torch.arange(yc1, dtype=torch.int64) % 2)
        .reshape(1, yc1)
        .expand(yr1, yc1)
        .contiguous()
    )
    add_inputs = (ya0, ya1, yidx0, ysrc0, yb0, yb1, yidx1)
    add_module = ScatterAddTestPreproc()
    add_results = [o.clone() for o in add_module(*add_inputs)]
    torch.save(add_results, os.path.join(output_dir, "scatter_add_test_results.pt"))
    with torch.no_grad():
        add_ep = torch.export.export(add_module, add_inputs, strict=False)
    add_pt2 = os.path.join(output_dir, "scatter_add_test.pt2")
    torch.export.save(add_ep, add_pt2)
    print(f"scatter_add: {len(add_ep.graph.nodes)} nodes -> {add_pt2}")

    # scatter_add feeding an elementwise consumer under a cooperative grid. The
    # accumulator is [S] while src/index are [N] (N = 16 * S), so a consumer
    # wrongly sized from the scatter's operands is off by 16x -- a shape
    # divergence, not just bad values. Every destination is hit 16 times, so the
    # atomic add is exercised; int64 keeps the parallel sum exact. 'x' straddles
    # both clamp bounds so neither clamp is a no-op.
    cg_s, cg_n = 256, 4096
    cg_base_a = torch.arange(cg_s, dtype=torch.int64)
    cg_base_b = torch.full((cg_s,), 7, dtype=torch.int64)
    cg_index = torch.arange(cg_n, dtype=torch.int64) % cg_s
    cg_src = torch.arange(cg_n, dtype=torch.int64) % 97
    cg_x = torch.arange(cg_s, dtype=torch.int64) * 100
    cg_flags = (torch.arange(cg_n, dtype=torch.int64) % 3) == 0
    cg_inputs = (cg_base_a, cg_base_b, cg_index, cg_src, cg_x, cg_flags)
    cg_module = ScatterAddCgConsumerTestPreproc()
    cg_results = [cg_module(*cg_inputs).clone()]
    torch.save(
        cg_results,
        os.path.join(output_dir, "scatter_add_cg_consumer_test_results.pt"),
    )
    with torch.no_grad():
        cg_ep = torch.export.export(cg_module, cg_inputs, strict=False)
    cg_pt2 = os.path.join(output_dir, "scatter_add_cg_consumer_test.pt2")
    torch.export.save(cg_ep, cg_pt2)
    print(f"scatter_add cg consumer: {len(cg_ep.graph.nodes)} nodes -> {cg_pt2}")

    # slice_scatter dim=1 multi-row repro. rows (64) > inner extent (32) so a
    # per-row failure shows. out0 contiguous src (control); out1 non-contiguous
    # src via a dim=1 view + clamp (the ROO pattern). Distinct arange values;
    # 'wide' scaled into [-8, 8] so clamp is a no-op and values stay distinct.
    dr, dc = 64, 33
    dbase0_a = torch.arange(dr * dc, dtype=torch.float32).reshape(dr, dc)
    dbase0_b = torch.zeros(dr * dc, dtype=torch.float32).reshape(dr, dc)
    dsrc0_a = (torch.arange(dr * 32, dtype=torch.float32) + 5000.0).reshape(dr, 32)
    dsrc0_b = torch.zeros(dr * 32, dtype=torch.float32).reshape(dr, 32)
    dbase1_a = (torch.arange(dr * dc, dtype=torch.float32) + 9000.0).reshape(dr, dc)
    dbase1_b = torch.zeros(dr * dc, dtype=torch.float32).reshape(dr, dc)
    dwide_a = (
        torch.arange(dr * dc, dtype=torch.float32) / (dr * dc) * 16.0 - 8.0
    ).reshape(dr, dc)
    dwide_b = torch.zeros(dr * dc, dtype=torch.float32).reshape(dr, dc)
    d1_inputs = (
        dbase0_a,
        dbase0_b,
        dsrc0_a,
        dsrc0_b,
        dbase1_a,
        dbase1_b,
        dwide_a,
        dwide_b,
    )
    d1_module = SliceScatterDim1ViewTestPreproc()
    d1_results = [o.clone() for o in d1_module(*d1_inputs)]
    torch.save(
        d1_results,
        os.path.join(output_dir, "slice_scatter_dim1_view_test_results.pt"),
    )
    with torch.no_grad():
        d1_ep = torch.export.export(d1_module, d1_inputs, strict=False)
    d1_pt2 = os.path.join(output_dir, "slice_scatter_dim1_view_test.pt2")
    torch.export.save(d1_ep, d1_pt2)
    print(f"slice_scatter dim1 view: {len(d1_ep.graph.nodes)} nodes -> {d1_pt2}")

    # slice_scatter with the open-ended `[:]` end sentinel (2**63-1). 1000 rows
    # spans many blocks and far exceeds the 18 columns, so the collapse onto
    # index 0 that a 32-bit 'end' produces is unmistakable. src values are
    # disjoint from the base's so a scatter that writes nothing is visible.
    orows, ocols = 1000, 18
    obase0_a = torch.arange(orows * ocols, dtype=torch.float32).reshape(orows, ocols)
    obase0_b = torch.zeros(orows * ocols, dtype=torch.float32).reshape(orows, ocols)
    osrc0 = (torch.arange(orows * ocols, dtype=torch.float32) + 1e6).reshape(
        orows, ocols
    )
    obase1_a = (torch.arange(orows * ocols, dtype=torch.float32) + 5000.0).reshape(
        orows, ocols
    )
    obase1_b = torch.zeros(orows * ocols, dtype=torch.float32).reshape(orows, ocols)
    osrc1 = (torch.arange(orows * (ocols - 2), dtype=torch.float32) + 2e6).reshape(
        orows, ocols - 2
    )
    oe_inputs = (obase0_a, obase0_b, osrc0, obase1_a, obase1_b, osrc1)
    oe_module = SliceScatterOpenEndTestPreproc()
    oe_results = [o.clone() for o in oe_module(*oe_inputs)]
    torch.save(
        oe_results,
        os.path.join(output_dir, "slice_scatter_open_end_test_results.pt"),
    )
    with torch.no_grad():
        oe_ep = torch.export.export(oe_module, oe_inputs, strict=False)
    oe_pt2 = os.path.join(output_dir, "slice_scatter_open_end_test.pt2")
    torch.export.save(oe_ep, oe_pt2)
    print(f"slice_scatter open end: {len(oe_ep.graph.nodes)} nodes -> {oe_pt2}")


if __name__ == "__main__":
    main()
