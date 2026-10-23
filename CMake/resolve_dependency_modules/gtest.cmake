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

# Keep this the same as ARROW_GTEST_BUILD_VERSION in the bundled Arrow's
# cpp/thirdparty/versions.txt. Without a system GoogleTest, the bundled Arrow
# builds libarrow_testing with that version, and Velox tests link
# libarrow_testing with this one.
set(VELOX_GTEST_VERSION 1.18.0)
set(
  VELOX_GTEST_BUILD_SHA256_CHECKSUM
  6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
)
set(
  VELOX_GTEST_SOURCE_URL
  "https://github.com/google/googletest/archive/refs/tags/v${VELOX_GTEST_VERSION}.tar.gz"
)

velox_resolve_dependency_url(GTEST)

message(STATUS "Building gtest from source")
FetchContent_Declare(
  googletest
  URL ${VELOX_GTEST_SOURCE_URL}
  URL_HASH ${VELOX_GTEST_BUILD_SHA256_CHECKSUM}
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(googletest)

# Mask compilation warning in clang 16.
target_compile_options(gtest PRIVATE -Wno-implicit-int-float-conversion)
