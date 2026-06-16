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

"""Fetch all review-relevant information for a GitHub PR in one shot.

Usage: fetch.py <owner/repo> <pr-number>
       fetch.py <github-pr-url>

Examples:
    fetch.py facebookincubator/velox 17495
    fetch.py https://github.com/facebookincubator/velox/pull/17495
"""

import json
import re
import subprocess
import sys


def run_gh(*args):
    result = subprocess.run(["gh"] + list(args), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        return None
    return result.stdout.strip()


def parse_args():
    if len(sys.argv) == 2:
        match = re.match(r"https?://github\.com/([^/]+/[^/]+)/pull/(\d+)", sys.argv[1])
        if match:
            return match.group(1), match.group(2)
        print(f"Invalid URL: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)
    elif len(sys.argv) == 3:
        return sys.argv[1], sys.argv[2]
    else:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)


def fetch_pr_metadata(repo, pr):
    raw = run_gh(
        "pr",
        "view",
        pr,
        "--repo",
        repo,
        "--json",
        "title,body,author,state,files,additions,deletions,baseRefName,headRefName",
    )
    if not raw:
        return None
    return json.loads(raw)


def fetch_diff(repo, pr):
    return run_gh("pr", "diff", pr, "--repo", repo)


def fetch_issue_comments(repo, pr):
    raw = run_gh(
        "api",
        f"repos/{repo}/issues/{pr}/comments",
        "--paginate",
    )
    if not raw:
        return []
    return json.loads(raw)


def fetch_review_comments(repo, pr):
    raw = run_gh(
        "api",
        f"repos/{repo}/pulls/{pr}/comments",
        "--paginate",
    )
    if not raw:
        return []
    return json.loads(raw)


def fetch_reviews(repo, pr):
    raw = run_gh(
        "api",
        f"repos/{repo}/pulls/{pr}/reviews",
    )
    if not raw:
        return []
    return json.loads(raw)


def print_section(title, content):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")
    print(content)


def main():
    repo, pr = parse_args()

    metadata = fetch_pr_metadata(repo, pr)
    if not metadata:
        sys.exit(1)

    print_section(
        f"PR #{pr}: {metadata['title']}",
        f"Author: {metadata['author']['login']}\n"
        f"State: {metadata['state']}\n"
        f"Branch: {metadata['headRefName']} -> {metadata['baseRefName']}\n"
        f"+{metadata['additions']} -{metadata['deletions']}\n"
        f"Files: {', '.join(f['path'] for f in metadata['files'])}\n"
        f"\n{metadata['body']}",
    )

    diff = fetch_diff(repo, pr)
    if diff:
        print_section("Diff", diff)

    skip = {"netlify[bot]", "github-actions[bot]"}

    comments = fetch_issue_comments(repo, pr)
    visible = [c for c in comments if c["user"]["login"] not in skip]
    if visible:
        lines = []
        for c in visible:
            lines.append(f"--- {c['user']['login']} ({c['created_at']}) ---")
            lines.append(c["body"])
            lines.append("")
        print_section("Comments", "\n".join(lines))

    reviews = fetch_reviews(repo, pr)
    visible_reviews = [r for r in reviews if r.get("body")]
    if visible_reviews:
        lines = []
        for r in visible_reviews:
            lines.append(
                f"--- {r['user']['login']} ({r['state']}, {r['submitted_at']}) ---"
            )
            lines.append(r["body"])
            lines.append("")
        print_section("Reviews", "\n".join(lines))

    inline = fetch_review_comments(repo, pr)
    if inline:
        lines = []
        for c in inline:
            loc = f"{c['path']}:{c.get('line', c.get('original_line', '?'))}"
            lines.append(f"--- {c['user']['login']} on {loc} ({c['created_at']}) ---")
            lines.append(c["body"])
            lines.append("")
        print_section("Inline Comments", "\n".join(lines))


if __name__ == "__main__":
    main()
