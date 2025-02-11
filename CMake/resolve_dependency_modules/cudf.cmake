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

set(VELOX_cudf_VERSION 25.04)
set(VELOX_cudf_BUILD_SHA256_CHECKSUM
    076bb16bde78927d7d8eed34ce102890bfc2f74896fea4dd90020bacb9a07f6b)
set(VELOX_cudf_SOURCE_URL
    "https://github.com/rapidsai/cudf/archive/1a891e6cfd1daef5bb56990cd18b4e3c7640fb53.tar.gz"
)
velox_resolve_dependency_url(cudf)

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
# Setup libcudf build to not have testing components
set(BUILD_TESTS OFF)
set(CUDF_BUILD_TESTUTIL OFF)
set(BUILD_SHARED_LIBS ON)

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
unset(BUILD_SHARED_LIBS)
endblock()
