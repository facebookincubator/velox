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
# docker-compose run -e NUM_THREADS=$(nproc) --rm ubuntu-cuda-cpp /bin/bash
# Then invoke ./build.sh to build with GPU support and run tests.

# Run a GPU build and test
pushd "$(dirname ${0})"

mkdir -p benchmark_results

queries=${1:-$(seq 1 22)}
devices=${2:-"cpu gpu"}
profile=${3:-"false"}

num_drivers=${NUM_DRIVERS:-16}
output_batch_rows=${BATCH_SIZE_ROWS:-100000}

for query_number in ${queries}; do
    printf -v query_number '%02d' "${query_number}"
    for device in ${devices}; do
        case "${device}" in
            "cpu")
                export VELOX_CUDF_DISABLED=1;;
            "gpu")
                export VELOX_CUDF_MEMORY_RESOURCE="async"
                export VELOX_CUDF_DISABLED=0;;
        esac
        echo "Running query ${query_number} on ${device} with ${num_drivers} drivers."
        # The benchmarks segfault after reporting results, so we disable errors
        PROFILE_CMD=""
        if [[ "${profile}" == "true" ]]; then
            PROFILE_CMD="nsys profile -t nvtx,cuda,osrt -f true --cuda-memory-usage=true --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --output=benchmark_results/q${query_number}_${device}_${num_drivers}_drivers.nsys-rep"
            # Enable GPU metrics if supported (Ampere or newer)
            if [[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | cut -d '.' -f 1)" -gt 7 ]]; then
                PROFILE_CMD="${PROFILE_CMD} --gpu-metrics-devices=0"
            fi
        fi

        set +e -x
        ${PROFILE_CMD} \
        ./_build/release/velox/benchmarks/tpch/velox_tpch_benchmark \
            --data_path=velox-tpch-sf10-data \
            --data_format=parquet \
            --run_query_verbose=${query_number} \
            --num_repeats=1 \
            --num_drivers=${num_drivers} \
            --preferred_output_batch_rows=${output_batch_rows} \
            --max_output_batch-rows=${output_batch_rows} 2>&1 \
            | tee benchmark_results/q${query_number}_${device}_${num_drivers}_drivers
        { set -e +x; } &> /dev/null
    done
done

popd
