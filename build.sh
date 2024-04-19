#!/bin/bash

set -euo pipefail

# Run this to launch the CUDA container:
# docker-compose run -e NUM_THREADS=$(nproc) --rm ubuntu-cuda-cpp
# Then invoke ./build.sh to build with GPU support and run tests.

# Run a GPU build and test
pushd "$(dirname ${0})"

#make cmake-gpu
make build

cd _build/release

ctest -R cudf -V

popd
