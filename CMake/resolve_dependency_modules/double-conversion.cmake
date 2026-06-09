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
include_guard(GLOBAL)

set(VELOX_DOUBLE_CONVERSION_VERSION v3.3.1)
set(
  VELOX_DOUBLE_CONVERSION_BUILD_SHA256_CHECKSUM
  fe54901055c71302dcdc5c3ccbe265a6c191978f3761ce1414d0895d6b0ea90e
)
set(
  VELOX_DOUBLE_CONVERSION_SOURCE_URL
  "https://github.com/google/double-conversion/archive/refs/tags/${VELOX_DOUBLE_CONVERSION_VERSION}.tar.gz"
)

velox_resolve_dependency_url(DOUBLE_CONVERSION)

message(STATUS "Building double-conversion from source")
FetchContent_Declare(
  double-conversion
  URL ${VELOX_DOUBLE_CONVERSION_SOURCE_URL}
  URL_HASH ${VELOX_DOUBLE_CONVERSION_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(double-conversion)

# Folly's bundled FindDoubleConversion.cmake uses find_path/find_library, which
# can't locate the in-tree headers or the to-be-built library at configure time.
# Pre-populate DOUBLE_CONVERSION_INCLUDE_DIR and DOUBLE_CONVERSION_LIBRARY so
# folly's find_package(DoubleConversion MODULE REQUIRED) succeeds and links
# against the bundled CMake target.
FetchContent_GetProperties(double-conversion SOURCE_DIR DOUBLE_CONVERSION_SOURCE_DIR)
set(
  DOUBLE_CONVERSION_INCLUDE_DIR
  ${DOUBLE_CONVERSION_SOURCE_DIR}
  CACHE PATH
  "double-conversion include directory"
  FORCE
)
set(
  DOUBLE_CONVERSION_LIBRARY
  double-conversion::double-conversion
  CACHE STRING
  "double-conversion library target"
  FORCE
)
