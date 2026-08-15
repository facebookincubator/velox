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

# Report build status and create failure metadata artifact.
#
# Usage: report-build-status.sh <job-name>
#
# Required environment variables:
#   BUILD_OUTCOME         - build step outcome (success/failure/cancelled)
#   BUILD_FAILURE_DETAILS - compiler errors extracted from build output (optional)

set -euo pipefail

JOB_NAME="$1"

if [[ -z $BUILD_OUTCOME || $BUILD_OUTCOME == "cancelled" ]]; then
  echo "Build was cancelled (likely superseded by a newer push). Skipping."
  exit 0
fi

if [[ $BUILD_OUTCOME != "success" ]]; then
  echo "::error::${JOB_NAME} build failed. Do not land this PR until the build is fixed."

  if [[ -n ${BUILD_FAILURE_DETAILS:-} ]]; then
    echo ""
    echo "Build errors:"
    echo "----------------------------------------"
    echo "$BUILD_FAILURE_DETAILS"
    echo "----------------------------------------"
  fi

  # Write failure metadata for the CI failure analysis workflow.
  mkdir -p /tmp/ci-failure
  jq -n --arg job "$JOB_NAME" --arg type "build" \
    '{job: $job, type: $type}' \
    >/tmp/ci-failure/failure.json

  exit 1
fi

echo "Build succeeded."
