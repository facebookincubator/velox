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

queries=${1:-$(seq 1 20)}
devices=${2:-"cpu gpu"}


for query_number in ${queries}; do
    printf -v query_number '%02d' "${query_number}"
    for device in ${devices}; do
        case "${device}" in
            "cpu")
                num_drivers=4
                export VELOX_CUDF_DISABLED=1;;
            "gpu")
                num_drivers=4
                export VELOX_CUDF_MEMORY_RESOURCE="async"
                export VELOX_CUDF_DISABLED=0;;
        esac
        echo "Running query ${query_number} on ${device} with ${num_drivers} drivers."
        # The benchmarks segfault after reporting results, so we disable errors
        set +e
        ./_build/release/velox/benchmarks/tpch/velox_tpch_benchmark --data_path=velox-tpch-sf10-data --data_format=parquet --run_query_verbose=${query_number} --num_repeats=1 --num_drivers ${num_drivers} 2>&1 | tee benchmark_results/q${query_number}_${device}_${num_drivers}_drivers
        set -e
    done
done

popd
