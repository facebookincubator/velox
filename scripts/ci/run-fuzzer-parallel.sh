#!/bin/bash
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

# Run multiple fuzzer instances in parallel with different seeds.
#
# Usage: run-fuzzer-parallel.sh <num_instances> <repro_base_dir> <binary> [args...]
#
# In the fuzzer arguments, use these placeholders which are replaced per instance:
#   __SEED__         -> unique random seed per instance
#   __INSTANCE_ID__  -> instance number (1, 2, ...)
#   __REPRO_DIR__    -> per-instance repro directory (<repro_base_dir>/instance_N)
#   __LOG_DIR__      -> per-instance log directory (<repro_base_dir>/instance_N/logs)

set -euo pipefail

NUM_INSTANCES=$1
REPRO_BASE=$2
BINARY=$3
shift 3

PIDS=()
FAILED=0

for i in $(seq 1 "$NUM_INSTANCES"); do
  REPRO_DIR="${REPRO_BASE}/instance_${i}"
  LOG_DIR="${REPRO_DIR}/logs"
  mkdir -p "${LOG_DIR}"
  SEED=$((RANDOM * 32768 + RANDOM + i))

  # Replace placeholders in arguments
  INSTANCE_ARGS=()
  for arg in "$@"; do
    arg="${arg//__SEED__/$SEED}"
    arg="${arg//__INSTANCE_ID__/$i}"
    arg="${arg//__REPRO_DIR__/$REPRO_DIR}"
    arg="${arg//__LOG_DIR__/$LOG_DIR}"
    INSTANCE_ARGS+=("$arg")
  done

  echo "Starting instance ${i}: seed=${SEED}, repro=${REPRO_DIR}"
  "$BINARY" "${INSTANCE_ARGS[@]}" >"${REPRO_DIR}/stdout.log" 2>&1 &
  PIDS+=($!)
done

echo "Waiting for ${NUM_INSTANCES} instances to complete..."

for i in "${!PIDS[@]}"; do
  IDX=$((i + 1))
  if ! wait "${PIDS[$i]}"; then
    echo "::error::Fuzzer instance ${IDX} FAILED (PID ${PIDS[$i]})"
    echo "=== Instance ${IDX} stdout (last 100 lines) ==="
    tail -100 "${REPRO_BASE}/instance_${IDX}/stdout.log" || true
    echo "=== End instance ${IDX} ==="
    FAILED=$((FAILED + 1))
  else
    echo "Instance ${IDX} passed"
  fi
done

if [ "$FAILED" -gt 0 ]; then
  echo "${FAILED} of ${NUM_INSTANCES} instance(s) failed"
  exit 1
fi

echo "All ${NUM_INSTANCES} instances passed"
