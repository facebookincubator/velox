#!/usr/bin/env bash
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

# Ensures the "Recent blog posts" section in README.md lists the 3 most recent
# blog posts from website/blog/. Auto-fixes if out of date.

set -euo pipefail

BLOG_DIR="website/blog"
NUM_POSTS=3

# Build expected blog list lines from the most recent .mdx files.
EXPECTED_LINES=()
for f in $(ls -1 "$BLOG_DIR"/*.mdx | sort -r | head -$NUM_POSTS); do
  DATE=$(basename "$f" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}')
  TITLE=$(grep '^title:' "$f" | sed 's/title: *//; s/"//g')
  SLUG=$(grep '^slug:' "$f" | sed 's/slug: *//')
  EXPECTED_LINES+=("- [$TITLE](https://velox-lib.io/blog/$SLUG) ($DATE)")
done

# Extract current blog list: everything between "Recent blog posts" header and
# the next markdown section (##).
CURRENT=$(sed -n '/^Recent blog posts/,/^##/{/^Recent blog posts/d;/^##/d;/^$/d;p;}' README.md)
EXPECTED=$(printf '%s\n' "${EXPECTED_LINES[@]}")

if [ "$CURRENT" = "$EXPECTED" ]; then
  exit 0
fi

# Auto-fix: replace everything between "Recent blog posts" header and the next
# section with the expected blog list.
TEMP=$(mktemp)
IN_SECTION=false
while IFS= read -r line; do
  if [[ $line == "Recent blog posts"* ]]; then
    echo "$line" >>"$TEMP"
    echo "" >>"$TEMP"
    printf '%s\n' "${EXPECTED_LINES[@]}" >>"$TEMP"
    IN_SECTION=true
    continue
  fi
  if $IN_SECTION; then
    # Skip until the next section header.
    if [[ $line == "##"* ]]; then
      IN_SECTION=false
      echo "" >>"$TEMP"
      echo "$line" >>"$TEMP"
    fi
    continue
  fi
  echo "$line" >>"$TEMP"
done <README.md
mv "$TEMP" README.md

echo "Updated recent blog posts in README.md"
exit 1
