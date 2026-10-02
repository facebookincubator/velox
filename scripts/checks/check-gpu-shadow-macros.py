#!/usr/bin/env python3
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
"""Holds the GPU shadow headers to the macro list of the headers they shadow.

A shadow replaces a real header on the include path of a CUDA translation
unit, so a macro the real header defines and the shadow does not is a macro
that simply vanishes for device code. The failure modes are asymmetric:

  * A check macro that vanishes is a compile error at the first body that
    uses it -- annoying, and self-announcing.
  * A macro defined to do the wrong thing is a wrong answer -- silent.

Neither is acceptable, but the one this check exists to prevent is the third
case: the shadow covering exactly the macros today's registered functions
happen to use, so that the next function to be registered discovers the gap.
Upstream adds check macros from time to time, and nothing else notices.

Not wired into pre-commit on purpose. It is a GPU-specific invariant and
pre-commit runs for everyone, most of whom never build the cuDF backend. Run
it when touching a shadow or after merging upstream:

    python3 scripts/checks/check-gpu-shadow-macros.py .

The check is deliberately syntactic: it compares the set of `#define VELOX_*`
names, not their meanings. Meaning is the shadow compile test's job.
"""

import re
import sys
from pathlib import Path

# Each pair is (real header, shadow of it). Add a row when a shadow is added.
SHADOWED = [
    (
        "velox/common/base/Exceptions.h",
        "velox/experimental/cudf/functions/gpu_shadows/velox/common/base/Exceptions.h",
    ),
    (
        "velox/common/base/Status.h",
        "velox/experimental/cudf/functions/gpu_shadows/velox/common/base/Status.h",
    ),
]

# Names the shadow is allowed not to define, each with a reason. Keep this
# short; a growing list means the shadow is drifting rather than tracking.
EXEMPT = {
    # Declares host-only throw helpers for a type the device never constructs.
    "VELOX_DECLARE_CHECK_FAIL_TEMPLATES",
}

DEFINE = re.compile(r"^\s*#\s*define\s+(VELOX_[A-Z0-9_]+)")


def macros(path: Path) -> set[str]:
    names = set()
    for line in path.read_text().splitlines():
        match = DEFINE.match(line)
        if match:
            names.add(match.group(1))
    return names


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    failures = []

    for real_rel, shadow_rel in SHADOWED:
        real, shadow = root / real_rel, root / shadow_rel
        if not real.exists() or not shadow.exists():
            failures.append(f"{real_rel} or its shadow is missing")
            continue

        missing = macros(real) - macros(shadow) - EXEMPT
        if missing:
            failures.append(
                f"{shadow_rel}\n  does not define, and {real_rel} does:\n"
                + "".join(f"    {name}\n" for name in sorted(missing))
            )

    if failures:
        print("GPU shadow headers are missing macros from the headers they shadow.")
        print()
        for failure in failures:
            print(failure)
        print(
            "A macro the shadow does not define vanishes for device code. Define it\n"
            "in the shadow with the same error class the real one raises -- see the\n"
            "comment at the top of the Exceptions.h shadow -- or add it to EXEMPT in\n"
            "this script with a reason."
        )
        return 1

    total = sum(len(macros(root / real)) for real, _ in SHADOWED)
    print(f"All {total} shadowed macros are defined in their GPU shadows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
