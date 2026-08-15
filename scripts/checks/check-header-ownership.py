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
"""
CI check: verify every .h file under velox/ is listed in a CMakeLists.txt target.

Exits with code 1 if orphan headers are found.

Usage:
    python3 .github/scripts/check-header-ownership.py [velox_source_dir]

The velox_source_dir defaults to "velox" (relative to repo root).
"""

import os
import re
import sys

SKIP_DIRS = {
    "external",
    "experimental",
    "facebook",
    "public_tld",
    "python",
}

SKIP_PATHS = {
    "tpcds/gen/dsdgen/include",
    "tpch/gen/dbgen/include",
}


def collect_cmake_headers(velox_dir):
    """Walk CMakeLists.txt files and collect all .h filenames mentioned."""
    tracked = set()

    for root, _dirs, files in os.walk(velox_dir):
        if "CMakeLists.txt" not in files:
            continue

        rel_root = os.path.relpath(root, velox_dir)

        cmake_path = os.path.join(root, "CMakeLists.txt")
        with open(cmake_path) as f:
            content = f.read()

        # Find all .h references in any target call or HEADERS block
        for h in re.findall(r"([\w][\w/\-]*\.h)", content):
            # Resolve relative to CMakeLists.txt directory
            tracked.add(os.path.normpath(os.path.join(rel_root, h)))

    return tracked


def collect_fs_headers(velox_dir):
    """Walk filesystem for all .h files, respecting skip rules."""
    headers = set()

    for root, dirs, files in os.walk(velox_dir):
        rel_root = os.path.relpath(root, velox_dir)

        # Skip entire directories
        parts = rel_root.split(os.sep)
        if any(p in SKIP_DIRS for p in parts):
            dirs.clear()
            continue

        if any(rel_root.startswith(sp) for sp in SKIP_PATHS):
            dirs.clear()
            continue

        for f in files:
            if f.endswith(".h"):
                headers.add(os.path.normpath(os.path.join(rel_root, f)))

    return headers


def main():
    velox_dir = sys.argv[1] if len(sys.argv) > 1 else "velox"

    if not os.path.isdir(velox_dir):
        print(f"Error: {velox_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    tracked = collect_cmake_headers(velox_dir)
    fs_headers = collect_fs_headers(velox_dir)

    orphans = sorted(fs_headers - tracked)

    if orphans:
        print(
            f"Found {len(orphans)} header(s) not tracked by any CMakeLists.txt target:"
        )
        for h in orphans:
            print(f"  {h}")
        print()
        print("To fix, add each header to the appropriate CMakeLists.txt:")
        print("  - For velox_add_library() targets: add to the HEADERS list")
        print(
            "  - For test/benchmark/fuzzer targets: use velox_add_test_headers(<target> <header>)"
        )
        sys.exit(1)
    else:
        print(f"All {len(fs_headers)} headers are tracked by CMakeLists.txt targets.")
        sys.exit(0)


if __name__ == "__main__":
    main()
