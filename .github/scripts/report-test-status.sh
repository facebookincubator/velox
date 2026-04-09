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

# Report test status and create failure metadata artifact.
#
# Usage: report-test-status.sh <job-name> <config-name>
#
# Required environment variables:
#   BUILD_OUTCOME  - build step outcome (success/failure/cancelled)
#   TEST_OUTCOME   - test step outcome (success/failure/cancelled)
#   FLAKY          - "true" if tests were flaky (failed then passed on retry)
#   FAILED_TESTS    - newline-separated list of failed ctest targets (optional)
#   FAILED_CASES    - newline-separated list of failed gtest cases (optional)
#   FAILURE_DETAILS - gtest output for failing tests (optional)

set -euo pipefail

JOB_NAME="$1"
CONFIG_NAME="$2"

if [[ -z $BUILD_OUTCOME || $BUILD_OUTCOME == "cancelled" ]]; then
  echo "Run was cancelled (likely superseded by a newer push). Skipping."
  exit 0
fi

if [[ $BUILD_OUTCOME != "success" ]]; then
  echo "::error::Build failed — tests did not run. Check the '${JOB_NAME}' build job for compiler errors and build logs."
  exit 1
fi

if [[ -z $TEST_OUTCOME || $TEST_OUTCOME == "cancelled" ]]; then
  echo "Tests were cancelled (likely superseded by a newer push). Skipping."
  exit 0
fi

if [[ $TEST_OUTCOME != "success" ]]; then
  if [[ -n $FAILED_CASES ]]; then
    CASE_LIST=$(echo "$FAILED_CASES" | sed 's/^/  - /')
    CASE_COUNT=$(echo "$FAILED_CASES" | wc -l | tr -d ' ')
    echo "::error::${CASE_COUNT} test case(s) failed in the ${CONFIG_NAME} configuration. Failed: $(echo "$FAILED_CASES" | tr '\n' ', ' | sed 's/,$//'). To see the full test output with failure details, click the '${JOB_NAME}' job in the workflow run, then expand the 'Run Tests' step."
    echo ""
    echo "Failed test cases:"
    echo "$CASE_LIST"
  elif [[ -n $FAILED_TESTS ]]; then
    TEST_LIST=$(echo "$FAILED_TESTS" | sed 's/^/  - /')
    TEST_COUNT=$(echo "$FAILED_TESTS" | wc -l | tr -d ' ')
    echo "::error::${TEST_COUNT} test(s) failed in the ${CONFIG_NAME} configuration. Failed tests: $(echo "$FAILED_TESTS" | tr '\n' ', ' | sed 's/,$//'). To see the full test output with failure details, click the '${JOB_NAME}' job in the workflow run, then expand the 'Run Tests' step."
    echo ""
    echo "Failed tests:"
    echo "$TEST_LIST"
  else
    echo "::error::Tests failed in the ${CONFIG_NAME} configuration but no specific test names were captured. Check the 'Run Tests' step in the '${JOB_NAME}' job for details."
  fi
  if [[ -n ${FAILURE_DETAILS:-} ]]; then
    echo ""
    echo "Failure details:"
    echo "$FAILURE_DETAILS"
  fi
  echo ""
  echo "To investigate, click the '${JOB_NAME}' job in the workflow run, then expand the 'Run Tests' step for the full ctest output with assertion failures and stack traces."

  # Write failure metadata for the CI failure analysis workflow.
  mkdir -p /tmp/ci-failure
  if [[ -n $FAILED_CASES ]]; then
    TESTS="$FAILED_CASES"
  elif [[ -n $FAILED_TESTS ]]; then
    TESTS="$FAILED_TESTS"
  else
    TESTS=""
  fi
  jq -n --arg job "$JOB_NAME" \
    --arg type "test" \
    --arg failed_tests "$TESTS" \
    '{job: $job, type: $type, failed_tests: $failed_tests}' \
    >/tmp/ci-failure/failure.json

  exit 1
fi

if [[ $FLAKY == "true" ]]; then
  echo "::warning::Some tests were flaky (failed then passed on retry)."
fi
echo "All tests passed in ${CONFIG_NAME}."
