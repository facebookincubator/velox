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
from velox.experimental.torchwave.tests.copy_test import make_inputs, make_modules


def export_one(name: str, module: torch.nn.Module, inputs, output_dir: str) -> None:
    outs = module(*(t.clone() for t in inputs))
    if isinstance(outs, torch.Tensor):
        outs = (outs,)
    results = [o.clone() for o in outs]
    results_path = os.path.join(output_dir, f"{name}_results.pt")
    torch.save(results, results_path)
    print(f"{name}: {len(results)} output(s) -> {results_path}")

    # run_decompositions() functionalizes: the in-place `copy_` becomes the
    # functional `aten.copy` feeding a `select_scatter` / `slice_scatter`, which
    # is the shape the ROO preproc graphs arrive in and the one torchwave's
    # scatter rewrites expect. Plain export() leaves the `copy_` in place.
    with torch.no_grad():
        exported_program = torch.export.export(
            module,
            tuple(t.clone() for t in inputs),
            strict=False,
        ).run_decompositions({})
    targets = [
        str(n.target) for n in exported_program.graph.nodes if n.op == "call_function"
    ]
    print(f"{name}: {len(targets)} call_function nodes")
    for target in sorted(set(targets)):
        print(f"    {targets.count(target):3d}x {target}")

    pt2_path = os.path.join(output_dir, f"{name}.pt2")
    torch.export.save(exported_program, pt2_path)
    print(f"{name}: saved {pt2_path} ({os.path.getsize(pt2_path)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    parser.add_argument(
        "--output_dir",
        default=default_dir,
        help="Directory to write the copy_*_test.pt2 and *_results.pt files",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_inputs = make_inputs()
    for name, module in make_modules().items():
        export_one(name, module, all_inputs[name], args.output_dir)


if __name__ == "__main__":
    main()
