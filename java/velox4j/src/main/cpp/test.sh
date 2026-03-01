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

set -e
set -o pipefail
set -u

BASE_DIR=$(dirname "$0")
NUM_THREADS=$(nproc)
SOURCE_DIR=$BASE_DIR
BUILD_DIR=$BASE_DIR/build
BUILD_TEST_DIR=$BUILD_DIR/test

# Build C++ so libraries.
cmake -DCMAKE_BUILD_TYPE=Release -DVELOX4J_ENABLE_CCACHE=ON -DVELOX4J_BUILD_TESTING=ON -S "$SOURCE_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j "$NUM_THREADS"

# Run tests.
cd "$BUILD_TEST_DIR"
ctest -V
