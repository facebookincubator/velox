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

set -euo pipefail

# To get the data, copy from /datasets/velox-tpch-sf10-data to this repo:
# cp -r /datasets/velox-tpch-sf10-data .

# Run this to launch the CUDA container:
# docker-compose run -e NUM_THREADS=$(nproc) --rm adapters-cuda /bin/bash
# Then invoke ./build.sh to build with GPU support and run tests.

# Run a GPU build and test
pushd "$(dirname ${0})"

mkdir -p benchmark_results

queries=${1:-$(seq 1 22)}
devices=${2:-"cpu gpu"}
profile=${3:-"false"}

num_drivers=${NUM_DRIVERS:-4}
output_batch_rows=${BATCH_SIZE_ROWS:-100000}
cudf_chunk_read_limit=$((1024 * 1024 * 1024 * 1))
cudf_pass_read_limit=0
VELOX_CUDF_MEMORY_RESOURCE="async"

for query_number in ${queries}; do
  printf -v query_number '%02d' "${query_number}"
  for device in ${devices}; do
    case "${device}" in
    "cpu")
      num_drivers=${NUM_DRIVERS:-32}
      VELOX_CUDF_ENABLED=false
      ;;
    "gpu")
      VELOX_CUDF_ENABLED=true
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
      ./_build/release/velox/benchmarks/tpch/velox_tpch_benchmark \
      --data_path=velox-tpch-sf100-data \
      --data_format=parquet \
      --run_query_verbose=${query_number} \
      --num_repeats=1 \
      --velox_cudf_enabled=${VELOX_CUDF_ENABLED} \
      --velox_cudf_memory_resource=${VELOX_CUDF_MEMORY_RESOURCE} \
      --num_drivers=${num_drivers} \
      --preferred_output_batch_rows=${output_batch_rows} \
      --max_output_batch-rows=${output_batch_rows} \
      --cudf_chunk_read_limit=${cudf_chunk_read_limit} \
      --cudf_pass_read_limit=${cudf_pass_read_limit} 2>&1 |
      tee benchmark_results/q${query_number}_${device}_${num_drivers}_drivers
    { set -e +x; } &>/dev/null
  done
done

popd
