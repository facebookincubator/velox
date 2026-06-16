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

"""Post a GitHub PR review from a file.

Usage: post.py <owner/repo> <pr-number> <event> <body-file>
       post.py <github-pr-url> <event> <body-file>

Events: APPROVE, REQUEST_CHANGES, COMMENT

Examples:
    post.py facebookincubator/velox 17495 REQUEST_CHANGES /tmp/review.md
    post.py https://github.com/facebookincubator/velox/pull/17495 APPROVE /tmp/review.md
"""

import json
import re
import subprocess
import sys

VALID_EVENTS = {"APPROVE", "REQUEST_CHANGES", "COMMENT"}


def run_gh(*args, stdin=None):
    result = subprocess.run(
        ["gh"] + list(args),
        input=stdin,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def parse_args():
    if len(sys.argv) == 4:
        match = re.match(r"https?://github\.com/([^/]+/[^/]+)/pull/(\d+)", sys.argv[1])
        if match:
            return match.group(1), match.group(2), sys.argv[2], sys.argv[3]
        print(f"Invalid URL: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)
    elif len(sys.argv) == 5:
        return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    else:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)


def main():
    repo, pr, event, body_file = parse_args()

    event = event.upper()
    if event not in VALID_EVENTS:
        print(
            f"Invalid event: {event}. Must be one of: {', '.join(sorted(VALID_EVENTS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(body_file) as f:
        body = f.read()

    if not body.strip():
        print("Error: review body is empty", file=sys.stderr)
        sys.exit(1)

    payload = json.dumps({"event": event, "body": body, "comments": []})

    result = run_gh(
        "api",
        f"repos/{repo}/pulls/{pr}/reviews",
        "--method",
        "POST",
        "--input",
        "-",
        stdin=payload,
    )

    data = json.loads(result)
    print(data.get("html_url", "Review submitted"))


if __name__ == "__main__":
    main()
