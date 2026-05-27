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

import argparse
import json
import re
import sys
import os

import util


# clang-tidy doesn't support CUDA compiler flags and CUDA headers.
DEFAULT_EXCLUDE_DIRS = ["cudf/", "wave/"]


class Multimap(dict):
    def __setitem__(self, key, value):
        if key not in self:
            dict.__setitem__(self, key, [value])  # call super method to avoid recursion
        else:
            self[key].append(value)


def should_exclude(file, exclude_dirs):
    return any(exclude_dir in file for exclude_dir in exclude_dirs)


def get_exclude_dirs(exclude_dirs):
    return list(dict.fromkeys(DEFAULT_EXCLUDE_DIRS + (exclude_dirs or [])))


def is_supported_file(file, exclude_dirs):
    # Exclude files in exclude_dirs.
    # Exclude *-inl.h files: they are designed to be included from their corresponding header
    # and cannot be compiled as standalone translation units.
    return (
        re.match(r".*(\.cpp|\.h|\.hpp)$", file)
        and not should_exclude(file, exclude_dirs)
        and not file.endswith("-inl.h")
    )


def git_changed_lines(commit, exclude_dirs):
    file = ""
    changed_lines = Multimap()

    for line in util.run(f"git diff --text --unified=0 {commit}")[1].splitlines():
        line = line.rstrip("\n")
        fields = line.split()

        match = re.match(r"^\+\+\+ b/.*", line)
        if match:
            file = ""

        match = re.match(r"^\+\+\+ b/(.*(\.cpp|\.h|\.hpp))$", line)
        if match:
            matched_file = match.group(1)
            if is_supported_file(matched_file, exclude_dirs):
                file = matched_file

        match = re.match(r"^@@", line)
        if match and file != "" and len(fields) >= 3:
            lspan = fields[2].split(",")
            if len(lspan) <= 1:
                lspan.append("0")

            start_line = int(lspan[0])
            line_count = int(lspan[1])

            # Skip invalid line ranges (e.g., +0,0 from deleted files)
            if start_line > 0 or line_count > 0:
                changed_lines[file] = [start_line, start_line + line_count]

    return changed_lines


def check_output(output):
    return re.match(r"(^/.* warning: |^$)", output)


def tidy(args):
    exclude_dirs = get_exclude_dirs(args.exclude_dirs)
    files = util.input_files(args.files)
    files = [file for file in files if is_supported_file(file, exclude_dirs)]

    in_gha = os.environ.get("GITHUB_ACTIONS") is not None
    changed_lines = git_changed_lines(args.commit, exclude_dirs)
    filtered_files = [file for file in files if file in changed_lines]
    line_filter = json.dumps(
        [{"name": file, "lines": changed_lines[file]} for file in filtered_files]
    )
    lines = f"'--line-filter={line_filter}'"

    if len(filtered_files) == 0:
        return 0

    fix = "--fix" if args.fix == "fix" else ""

    ok = True
    build_path = args.p or os.getenv("BUILD_PATH")
    build_path_str = f"-p {build_path}" if build_path else ""

    if build_path_str == "" and not os.path.isfile(
        os.getcwd().join("compile_commands.json")
    ):
        print("compile_commands.json not found, skipping clang-tidy")
        return 0

    status, stdout, stderr = util.run(
        f"xargs clang-tidy --format-style=file -header-filter='.*' --quiet {build_path_str} {fix} {lines}",
        input=filtered_files,
    )

    if in_gha:
        clang_tidy_pattern = (
            r"^(.*):(\d+):(\d+):\s+(error|warning):\s+(.*) \[([a-z0-9,\-]+)\]\s*$"
        )

        for stdout_line in stdout.split("\n"):
            m = re.match(clang_tidy_pattern, stdout_line)
            if m is not None:
                file, line, col, severity, message, rule = m.groups()
                file = file.removeprefix("/__w/velox/velox/")
                print(
                    f"::{severity} file={file},line={line},col={col},title={rule}::{message}"
                )

    ok = check_output(stdout)
    if not ok:
        print(stdout)

    return 0 if ok else 1


def parse_args():
    global parser
    parser = argparse.ArgumentParser(description="Clang Tidy Utility")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--fix")
    parser.add_argument("-p", help="Path containing 'compile_commands.json'")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        dest="exclude_dirs",
        help="Additional path substring to exclude from clang-tidy.",
    )

    parser.add_argument("files", metavar="FILES", nargs="+", help="files to process")

    return parser.parse_args()


def main():
    return tidy(parse_args())


if __name__ == "__main__":
    sys.exit(main())
