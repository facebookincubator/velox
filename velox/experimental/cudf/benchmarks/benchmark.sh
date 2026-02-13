#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
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

set -euo pipefail

# Put the generated data in the directory velox-tpch-sf10-data at the root of this repo

# Run this to launch the CUDA container:
# docker-compose run -e NUM_THREADS=$(nproc) --rm adapters-cuda /bin/bash
# Then run the following to build with GPU support.
# CUDA_ARCHITECTURES=native EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_BENCHMARKS=ON" make cudf
# Then invoke this script from the root of the repo:
# ./velox/experimental/cudf/benchmarks/benchmark.sh

mkdir -p benchmark_results

queries=${1:-$(seq 1 22)}
devices=${2:-"cpu gpu"}
profile=${3:-"false"}
# Resolve script dir so binaries can be run from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

cudf_chunk_read_limit=$((1024 * 1024 * 1024 * 1))
cudf_pass_read_limit=0
VELOX_CUDF_MEMORY_RESOURCE="async"
VELOX_CUDF_MEMORY_PERCENT=0

for query_number in ${queries}; do
  printf -v query_number '%02d' "${query_number}"
  for device in ${devices}; do
    case "${device}" in
    "cpu")
      num_drivers=${NUM_DRIVERS:-32}
      BENCHMARK_EXECUTABLE="${REPO_ROOT}/_build/release/velox/benchmarks/tpch/velox_tpch_benchmark"
      CUDF_FLAGS=""
      ;;
    "gpu")
      num_drivers=${NUM_DRIVERS:-4}
      BENCHMARK_EXECUTABLE="${REPO_ROOT}/_build/release/velox/experimental/cudf/benchmarks/velox_cudf_tpch_benchmark"
      CUDF_FLAGS="--velox_cudf_table_scan=true --cudf_chunk_read_limit=${cudf_chunk_read_limit} --cudf_pass_read_limit=${cudf_pass_read_limit} --cudf_memory_resource=${VELOX_CUDF_MEMORY_RESOURCE} --cudf_memory_percent=${VELOX_CUDF_MEMORY_PERCENT}"
      ;;
    esac
    echo "Running query ${query_number} on ${device} with ${num_drivers} drivers."
    # The benchmarks segfault after reporting results, so we disable errors
    PROFILE_CMD=""
    if [[ ${profile} == "true" ]]; then
      PROFILE_CMD="nsys profile -t nvtx,cuda,osrt -f true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --output=benchmark_results/q${query_number}_${device}_${num_drivers}_drivers.nsys-rep"
      # Enable GPU metrics if supported (Ampere or newer)
      if [[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | cut -d '.' -f 1)" -gt 7 ]]; then
        device_id=${CUDA_VISIBLE_DEVICES:-"0"}
        PROFILE_CMD="${PROFILE_CMD} --gpu-metrics-devices=${device_id}"
      fi
    fi

    set +e -x
    ${PROFILE_CMD} \
      ${BENCHMARK_EXECUTABLE} \
      --data_path=velox-tpch-sf10-data \
      --data_format=parquet \
      --run_query_verbose=${query_number} \
      --num_repeats=1 \
      --num_drivers=${num_drivers} \
      ${CUDF_FLAGS} 2>&1 |
      tee benchmark_results/q${query_number}_${device}_${num_drivers}_drivers
    { set -e +x; } &>/dev/null
  done
done
