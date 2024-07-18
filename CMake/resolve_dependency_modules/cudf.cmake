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

set(VELOX_cudf_VERSION 24.06)
set(VELOX_cudf_BUILD_SHA256_CHECKSUM
    f318032d01d43e14214ed70b6013ee0581d0327be49b858c75644f4bfc5f694b)
set(VELOX_cudf_SOURCE_URL
    "https://github.com/rapidsai/cudf/archive/refs/tags/v24.06.01.tar.gz")
resolve_dependency_url(cudf)

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

# libcudf's `get_arrow.cmake` check for sentinal targets to determine if arrow
# is already part of the build graph. Use that to early terminate and allow us
# to use the existing external project arrow
#
# Check to make sure we didn't find an installed arrow
if(NOT TARGET arrow_static)
  set(CUDF_USE_ARROW_STATIC ON)
  add_library(arrow_static INTERFACE IMPORTED GLOBAL)
  target_link_libraries(arrow_static INTERFACE arrow)
endif()

FetchContent_Declare(
  cudf
  URL ${VELOX_cudf_SOURCE_URL}
  URL_HASH ${VELOX_cudf_BUILD_SHA256_CHECKSUM}
  SOURCE_SUBDIR cpp)

FetchContent_MakeAvailable(cudf)
endblock()

# Make sure we don't build cudf till arrow external project is finished
if(TARGET arrow_ep)
  add_dependencies(cudf arrow_ep)
endif()
