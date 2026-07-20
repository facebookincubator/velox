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

# Extract failed test names, gtest cases, and failure details from ctest output.
# Writes results to $GITHUB_OUTPUT as failed-tests, failed-cases, and
# failure-details.
#
# Usage: extract-test-failures.sh <ctest-log-path>

set -euo pipefail

CTEST_LOG="$1"

FAILED_TESTS=$(grep -A 1000 'The following tests FAILED:' "$CTEST_LOG" | grep -E '^\s+[0-9]+ - ' | sed 's/.*- \(.*\) (.*/\1/' | head -20 || true)
FAILED_CASES=$(grep -E '^\[  FAILED  \]' "$CTEST_LOG" | sed 's/\[  FAILED  \] //' | sed 's/ (.*//' | grep '\.' | sort -u | head -20 || true)
FAILURE_DETAILS=$(awk '/^\[ RUN      \]/{buf=$0; next} buf{buf=buf"\n"$0} /^\[  FAILED  \]/ && buf{print buf; buf=""} /^\[       OK \]/{buf=""}' "$CTEST_LOG" | head -200 || true)

if [[ -n $FAILED_TESTS ]]; then
  {
    echo 'failed-tests<<EOF'
    echo "$FAILED_TESTS"
    echo 'EOF'
  } >>"$GITHUB_OUTPUT"
fi

if [[ -n $FAILED_CASES ]]; then
  {
    echo 'failed-cases<<EOF'
    echo "$FAILED_CASES"
    echo 'EOF'
  } >>"$GITHUB_OUTPUT"
fi

if [[ -n $FAILURE_DETAILS ]]; then
  {
    echo 'failure-details<<EOF'
    echo "$FAILURE_DETAILS"
    echo 'EOF'
  } >>"$GITHUB_OUTPUT"
fi
