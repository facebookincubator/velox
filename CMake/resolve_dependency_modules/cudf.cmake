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

set(VELOX_cudf_VERSION 24.10)
set(VELOX_cudf_BUILD_SHA256_CHECKSUM
    daa270c1e9223f098823491606bad2d9b10577d4bea8e543ae80265f1cecc0ed)
set(VELOX_cudf_SOURCE_URL
    "https://github.com/rapidsai/cudf/archive/refs/tags/v24.10.01.tar.gz")
velox_resolve_dependency_url(cudf)

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
# Setup libcudf build to not have testing components
set(BUILD_TESTS OFF)
set(CUDF_BUILD_TESTUTIL OFF)

# cudf sets all warnings as errors, and therefore fails to compile with velox
# expanded set of warnings. We selectively disable problematic warnings just for
# cudf
string(APPEND CMAKE_CXX_FLAGS
       " -Wno-non-virtual-dtor -Wno-missing-field-initializers")
string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-copy")

set(fmt_scope_patch
    patch -p1 <
    ${CMAKE_CURRENT_SOURCE_DIR}/CMake/resolve_dependency_modules/fmt_scope.patch
)

FetchContent_Declare(
  cudf
  URL ${VELOX_cudf_SOURCE_URL}
  URL_HASH ${VELOX_cudf_BUILD_SHA256_CHECKSUM}
  SOURCE_SUBDIR cpp
  PATCH_COMMAND ${fmt_scope_patch}
  UPDATE_DISCONNECTED 1)

FetchContent_MakeAvailable(cudf)
endblock()
