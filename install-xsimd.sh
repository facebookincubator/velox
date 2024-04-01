#!/bin/bash
if [ ! -d "xsimd" ]; then
    git clone https://github.com/xtensor-stack/xsimd --recurse-submodules
fi
cd xsimd

mkdir -p build
cd build

cmake ../
cmake --build . --config=Debug
cmake --install . --config=Debug
