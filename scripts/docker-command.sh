#!/bin/bash

# Compilation and testing
make
cd _build/release && ctest -j${NUM_THREADS} -VV --output-on-failure

