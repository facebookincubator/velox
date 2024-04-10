#!/bin/bash

# Run this inside the CUDA container:
# docker-compose run -e NUM_THREADS=$(nproc) --rm ubuntu-cuda-cpp

# Run a GPU build and test
pushd "$(dirname ${0})"

make cmake-gpu && make build

pushd _build/release

ctest

popd

popd
