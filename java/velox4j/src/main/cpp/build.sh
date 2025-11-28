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
INSTALL_DESTINATION=$BUILD_DIR/dist/lib
VELOX4J_LIB_NAME=libvelox4j.so

# Build C++ so libraries.
cmake -DCMAKE_BUILD_TYPE=Release -DVELOX4J_ENABLE_CCACHE=ON -DVELOX4J_BUILD_TESTING=OFF -DVELOX4J_INSTALL_DESTINATION="$INSTALL_DESTINATION" -S "$SOURCE_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --target velox4j -j "$NUM_THREADS"
cmake --install "$BUILD_DIR" --component velox4j

# Do a check to exclude the case that CMake installation incorrectly installed symbolic links.
for file in "$INSTALL_DESTINATION"/*; do
  if [ -L "$file" ]; then
    echo "CMake installation just created a symbolic link $file in the installation directory, this is not expected, aborting..."
    exit 1
  fi
done

# Force '$ORIGIN' runpaths for all so libraries to make the build portable.
# 1. Remove any already set RUNPATH sections.
for file in "$INSTALL_DESTINATION"/*; do
  echo "Removing RUNPATH on file: $file ..."
  patchelf --remove-rpath "$file"
done

# 2. Add new RUNPATH sections with '$ORIGIN'.
for file in "$INSTALL_DESTINATION"/*; do
  echo "Adding RUNPATH on file: $file ..."
  patchelf --set-rpath '$ORIGIN' "$file"
done

# 3. Print new ELF headers.
for file in "$INSTALL_DESTINATION"/*; do
  echo "Checking ELF header on file: $file ..."
  readelf -d "$file"
done

# Do final checks.
# 1. Check ldd result.
echo "Checking ldd result of libvelox4j.so: "
ldd "$INSTALL_DESTINATION/$VELOX4J_LIB_NAME"

# 2. Check ld result.
echo "Checking ld result of libvelox4j.so: "
ld "$INSTALL_DESTINATION/$VELOX4J_LIB_NAME"

# Finished.
echo "Successfully built velox4j-cpp."
