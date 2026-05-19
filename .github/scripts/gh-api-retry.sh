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

# shellcheck shell=bash

# Source this file from a workflow `run:` step to define a `gh_api`
# function that wraps `gh api` with retry-with-exponential-backoff.
# Use it in place of bare `gh api` for any GitHub API call from CI:
# transient network blips, rate-limit hiccups (especially on the
# stricter /search/issues path), and 5xx upstream failures all become
# survivable instead of single-shot workflow failures.
#
# Args: same as `gh api`. Stdout is preserved on the successful
# attempt so jq pipelines still work. On a final failure, the function
# returns gh's exit code so the caller's `set -e` semantics are
# unchanged.
#
# Env knobs (sensible defaults; tweak only if a caller has a reason):
#   GH_API_RETRY_MAX     — max attempts, default 3
#   GH_API_RETRY_BACKOFF — base sleep seconds, default 2 (exponential:
#                          2, 4, 8 between attempts when base=2)
#
# Usage:
#   source .github/scripts/gh-api-retry.sh
#   gh_api "/repos/$OWNER/$REPO/pulls/$PR"
#   gh_api --paginate "/repos/$OWNER/$REPO/issues/$N/comments"
#   gh_api -X POST "/repos/$OWNER/$REPO/actions/runs/$RUN/rerun"
gh_api() {
  local max_attempts="${GH_API_RETRY_MAX:-3}"
  local sleep_base="${GH_API_RETRY_BACKOFF:-2}"
  local attempt=1
  local exit_code=0
  while [ "$attempt" -le "$max_attempts" ]; do
    if gh api "$@"; then
      return 0
    fi
    exit_code=$?
    if [ "$attempt" -lt "$max_attempts" ]; then
      local wait_secs=$((sleep_base ** attempt))
      echo "::warning::gh api attempt ${attempt}/${max_attempts} failed (exit ${exit_code}); retrying in ${wait_secs}s" >&2
      sleep "$wait_secs"
    fi
    attempt=$((attempt + 1))
  done
  echo "::error::gh api failed after ${max_attempts} attempts (final exit ${exit_code}): gh api $*" >&2
  return "$exit_code"
}
