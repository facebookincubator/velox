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

# Run this to launch the CUDA container:
# docker-compose run -e NUM_THREADS=$(nproc) --rm ubuntu-cuda-cpp /bin/bash
# Then invoke ./build.sh to build with GPU support and run tests.

# Run a GPU build and test
pushd "$(dirname ${0})"

CUDA_ARCHITECTURES="native" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_ARROW=ON -DVELOX_ENABLE_PARQUET=ON -DVELOX_ENABLE_BENCHMARKS=ON -DVELOX_ENABLE_BENCHMARKS_BASIC=ON" make gpu

cd _build/release

ctest -R cudf -V

popd
