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

"""Parse CMake File API codemodel output and generate a dependency graph JSON.

This script reads the CMake File API reply directory (codemodel-v2) and
produces a JSON file containing:
  - file_to_targets: mapping of source file paths to their owning targets
  - target_deps: mapping of each target to its direct dependencies
  - header_to_sources: mapping of header file paths to source files that
    include them (via g++ -MM scanning of compile_commands.json)

The total target count can be derived from len(target_deps).

Usage:
    python generate-dependency-graph.py \\
        --build-dir _build/release \\
        --source-dir . \\
        --output dependency-graph.json
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def find_codemodel_reply(reply_dir: Path) -> dict:
    """Find and parse the codemodel reply index file."""
    for f in sorted(reply_dir.iterdir()):
        if f.name.startswith("codemodel-v2-") and f.suffix == ".json":
            with open(f) as fp:
                return json.load(fp)
    print("ERROR: No codemodel-v2 reply found in", reply_dir, file=sys.stderr)
    sys.exit(1)


def parse_target_file(reply_dir: Path, target_json_file: str) -> dict:
    """Parse a single target JSON file from the file API reply."""
    target_path = reply_dir / target_json_file
    with open(target_path) as fp:
        return json.load(fp)


def build_dependency_graph(
    build_dir: str, source_dir: str
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build the dependency graph from CMake File API output.

    Returns:
        file_to_targets: maps relative source file paths to list of target names
        target_deps: maps each target name to its direct dependency target names
    """
    reply_dir = Path(build_dir) / ".cmake" / "api" / "v1" / "reply"
    if not reply_dir.exists():
        print("ERROR: File API reply directory not found:", reply_dir, file=sys.stderr)
        print(
            "Ensure cmake was configured with the file API query file.",
            file=sys.stderr,
        )
        sys.exit(1)

    source_dir = os.path.realpath(source_dir)
    build_dir_real = os.path.realpath(build_dir)

    codemodel = find_codemodel_reply(reply_dir)

    file_to_targets: dict[str, list[str]] = {}
    target_deps: dict[str, list[str]] = {}

    configurations = codemodel.get("configurations", [])
    if not configurations:
        print("ERROR: No configurations found in codemodel reply.", file=sys.stderr)
        sys.exit(1)

    config = configurations[0]
    targets = config.get("targets", [])

    # Build an ID-to-name lookup to avoid re-parsing target files for deps.
    id_to_name: dict[str, str] = {}
    target_data_cache: dict[str, dict] = {}
    for target_ref in targets:
        target_json_file = target_ref["jsonFile"]
        target_data = parse_target_file(reply_dir, target_json_file)
        target_id = target_ref.get("id", "")
        target_name = target_data["name"]
        id_to_name[target_id] = target_name
        target_data_cache[target_id] = target_data

    for target_ref in targets:
        target_id = target_ref.get("id", "")
        target_data = target_data_cache[target_id]
        target_name = id_to_name[target_id]

        # Extract source files.
        for source in target_data.get("sources", []):
            source_path = source.get("path", "")
            if not source_path:
                continue

            # Resolve to absolute then make relative to source dir.
            if not os.path.isabs(source_path):
                abs_path = os.path.normpath(os.path.join(source_dir, source_path))
            else:
                abs_path = os.path.normpath(source_path)

            # Skip generated files (those in the build directory).
            if abs_path.startswith(build_dir_real):
                continue

            # Make path relative to source directory.
            try:
                rel_path = os.path.relpath(abs_path, source_dir)
            except ValueError:
                continue

            # Skip paths outside the source tree.
            if rel_path.startswith(".."):
                continue

            if rel_path not in file_to_targets:
                file_to_targets[rel_path] = []
            if target_name not in file_to_targets[rel_path]:
                file_to_targets[rel_path].append(target_name)

        # Extract dependencies.
        dep_names = []
        for dep in target_data.get("dependencies", []):
            dep_id = dep.get("id", "")
            if dep_id in id_to_name:
                dep_names.append(id_to_name[dep_id])
        target_deps[target_name] = dep_names

    return file_to_targets, target_deps


def extract_flags(command: str) -> list[str]:
    """Extract include paths, defines, and std flag from a compile command."""
    flags = []
    try:
        tokens = shlex.split(command)
    except ValueError:
        return flags

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-I", "-isystem", "-D") and i + 1 < len(tokens):
            flags.extend([tok, tokens[i + 1]])
            i += 2
        elif tok.startswith(("-I", "-isystem", "-D")):
            flags.append(tok)
            i += 1
        elif re.match(r"-std=", tok):
            flags.append(tok)
            i += 1
        else:
            i += 1
    return flags


def scan_one_file(
    entry: dict,
) -> tuple[str, list[str]]:
    """Run g++ -MM -MG on a single compile_commands.json entry.

    Returns (source_file_path, [header_paths]) or (source_file_path, [])
    on failure.
    """
    source_file = entry["file"]
    directory = entry.get("directory", ".")
    command = entry.get("command", "")
    if not command:
        args = entry.get("arguments", [])
        command = " ".join(shlex.quote(a) for a in args)

    flags = extract_flags(command)
    cmd = ["g++", "-MM", "-MG"] + flags + [source_file]

    try:
        result = subprocess.run(
            cmd,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  WARNING: g++ -MM failed for {source_file}: {e}", file=sys.stderr)
        return source_file, []

    if result.returncode != 0:
        return source_file, []

    # Parse make-style output: target.o: source.cpp header1.h header2.h ...
    output = result.stdout.replace("\\\n", " ")
    colon_idx = output.find(":")
    if colon_idx == -1:
        return source_file, []

    deps_str = output[colon_idx + 1 :]
    deps = deps_str.split()
    # First entry is the source file itself; remaining are headers.
    headers = deps[1:] if len(deps) > 1 else []
    return source_file, headers


def scan_header_deps(
    build_dir: str, source_dir: str
) -> dict[str, list[str]]:
    """Scan compile_commands.json with g++ -MM to build header_to_sources map."""
    compile_commands_path = os.path.join(build_dir, "compile_commands.json")
    if not os.path.exists(compile_commands_path):
        print(
            "WARNING: compile_commands.json not found, skipping header scan.",
            file=sys.stderr,
        )
        return {}

    with open(compile_commands_path) as fp:
        entries = json.load(fp)

    print(f"  Scanning {len(entries)} compilation units with g++ -MM ...")
    source_dir_real = os.path.realpath(source_dir)
    build_dir_real = os.path.realpath(build_dir)

    header_to_sources: dict[str, set[str]] = defaultdict(set)
    workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(scan_one_file, entries)
        scanned = 0
        failed = 0
        for source_file, headers in results:
            if not headers:
                failed += 1
                continue
            scanned += 1

            # Normalize source path relative to source dir.
            if os.path.isabs(source_file):
                source_abs = os.path.normpath(source_file)
            else:
                # Resolve relative to the entry's directory — but we
                # don't have it here.  The source_file in
                # compile_commands.json is typically absolute.
                source_abs = os.path.normpath(
                    os.path.join(source_dir_real, source_file)
                )

            if source_abs.startswith(build_dir_real):
                continue
            try:
                rel_source = os.path.relpath(source_abs, source_dir_real)
            except ValueError:
                continue
            if rel_source.startswith(".."):
                continue

            for header in headers:
                header_abs = os.path.normpath(header)
                if not os.path.isabs(header_abs):
                    continue
                if header_abs.startswith(build_dir_real):
                    continue
                try:
                    rel_header = os.path.relpath(header_abs, source_dir_real)
                except ValueError:
                    continue
                if rel_header.startswith(".."):
                    continue
                header_to_sources[rel_header].add(rel_source)

    # Convert sets to sorted lists for JSON serialization.
    result = {k: sorted(v) for k, v in header_to_sources.items()}
    print(f"  Header scan complete: {scanned} succeeded, {failed} failed")
    print(f"  Headers mapped: {len(result)}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate dependency graph from CMake File API output."
    )
    parser.add_argument(
        "--build-dir",
        required=True,
        help="Path to the CMake build directory.",
    )
    parser.add_argument(
        "--source-dir",
        default=".",
        help="Path to the source directory (default: current directory).",
    )
    parser.add_argument(
        "--output",
        default="dependency-graph.json",
        help="Output JSON file path (default: dependency-graph.json).",
    )
    args = parser.parse_args()

    file_to_targets, target_deps = build_dependency_graph(
        args.build_dir, args.source_dir
    )

    # Phase 2: Scan header dependencies via g++ -MM.
    header_to_sources = scan_header_deps(args.build_dir, args.source_dir)

    graph = {
        "file_to_targets": file_to_targets,
        "header_to_sources": header_to_sources,
        "target_deps": target_deps,
    }

    with open(args.output, "w") as fp:
        json.dump(graph, fp, indent=2, sort_keys=True)

    print(f"Dependency graph written to {args.output}")
    print(f"  Files mapped: {len(file_to_targets)}")
    print(f"  Headers mapped: {len(header_to_sources)}")
    print(f"  Targets: {len(target_deps)}")


if __name__ == "__main__":
    main()
