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

# Parse action argument: must be -check or -fix
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 -check | -fix"
  exit 1
fi

ACTION="$1"

if [[ $ACTION != "-check" && $ACTION != "-fix" ]]; then
  echo "Invalid argument: $ACTION"
  echo "Usage: $0 -check | -fix"
  exit 1
fi

# Ensure Maven is installed.
mvn -version

# Directory containing the source code to check
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../../../"

# Set image name.
IMAGE_NAME=velox4j-format

# Set Ubuntu version.
OS_VERSION=24.04

# Build the Docker image, passing the OS_VERSION build argument.
docker build --build-arg "OS_VERSION=$OS_VERSION" -t "$IMAGE_NAME:$OS_VERSION" "$SCRIPT_DIR"

# Determine the clang-format command
if [[ $ACTION == "-check" ]]; then
  FORMAT_COMMAND="clang-format-18 --dry-run --Werror"
  CMAKE_FORMAT_COMMAND="cmake-format --first-comment-is-literal True --check"
  MAVEN_COMMAND="spotless:check"
else
  FORMAT_COMMAND="clang-format-18 -i"
  CMAKE_FORMAT_COMMAND="cmake-format --first-comment-is-literal True -i"
  MAVEN_COMMAND="spotless:apply"
fi

# CPP code path.
CPP_ROOT="$SRC_DIR/src/main/cpp/"
CPP_MAIN_DIR="main/"
CPP_TEST_DIR="test/"

# 1. Run clang-format-18 on the CPP main code.
docker run --rm -v "$CPP_ROOT":/workspace -w /workspace "$IMAGE_NAME:$OS_VERSION" \
  sh -c "find $CPP_MAIN_DIR \( -name '*.h' -o -name '*.cc' -o -name '*.cpp' \) | xargs -r $FORMAT_COMMAND"

# 2. Run clang-format-18 on the CPP test code.
docker run --rm -v "$CPP_ROOT":/workspace -w /workspace "$IMAGE_NAME:$OS_VERSION" \
  sh -c "find $CPP_TEST_DIR \( -name '*.h' -o -name '*.cc' -o -name '*.cpp' \) | xargs -r $FORMAT_COMMAND"

# 3. Run cmake-format on root CMakeLists.txt.
docker run --rm -v "$CPP_ROOT":/workspace -w /workspace "$IMAGE_NAME:$OS_VERSION" \
  sh -c "$CMAKE_FORMAT_COMMAND CMakeLists.txt"

# 4. Run cmake-format on CPP main code.
docker run --rm -v "$CPP_ROOT":/workspace -w /workspace "$IMAGE_NAME:$OS_VERSION" \
  sh -c "find $CPP_MAIN_DIR \( -name 'CMakeLists.txt' -o -name '*.cmake' \) | xargs -r $CMAKE_FORMAT_COMMAND"

# 5. Run cmake-format on CPP test code.
docker run --rm -v "$CPP_ROOT":/workspace -w /workspace "$IMAGE_NAME:$OS_VERSION" \
  sh -c "find $CPP_TEST_DIR \( -name 'CMakeLists.txt' -o -name '*.cmake' \) | xargs -r $CMAKE_FORMAT_COMMAND"

# 6. Run Maven Spotless check or apply under the project root.
(
  cd "$SRC_DIR"
  mvn "$MAVEN_COMMAND"
)
