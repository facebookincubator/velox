#!/bin/bash

# Run a GPU build and test
pushd "$(dirname ${0})"

make cmake-gpu && make build

pushd _build/release

ctest

popd

popd
