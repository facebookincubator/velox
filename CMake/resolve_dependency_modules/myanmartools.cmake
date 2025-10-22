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

# Myanmar-tools is a C++ library for detecting and converting Zawgyi-encoded text
# Source: https://github.com/google/myanmar-tools
# Using commit f7efa5761c6aea3f6ab029600c0d296e16bed14c which corresponds to Java v1.1.3 release
# Note: Myanmar Tools uses language-specific tags (v1.1.3+java, v1.2.0+py, etc.)
# but doesn't have C++ specific version tags. We use the commit hash instead.
set(VELOX_MYANMARTOOLS_COMMIT f7efa5761c6aea3f6ab029600c0d296e16bed14c)
set(VELOX_MYANMARTOOLS_BUILD_SHA256_CHECKSUM
    cccd1217115e6d2eb54bb62a6e551727ebc44fa2ee518729775b6bb4eefb3e08)
set(VELOX_MYANMARTOOLS_SOURCE_URL
    "https://github.com/google/myanmar-tools/archive/${VELOX_MYANMARTOOLS_COMMIT}.tar.gz"
)

velox_resolve_dependency_url(MYANMARTOOLS)

message(STATUS "Building myanmar-tools from source")

FetchContent_Declare(
  myanmartools
  URL ${VELOX_MYANMARTOOLS_SOURCE_URL}
  URL_HASH SHA256=${VELOX_MYANMARTOOLS_BUILD_SHA256_CHECKSUM})

FetchContent_MakeAvailable(myanmartools)

# The myanmar-tools library builds targets like "myanmar-tools" or similar
# We need to check the actual CMakeLists.txt in the library to get the correct target name
# and create an alias
if(TARGET myanmartools)
  add_library(myanmartools::myanmartools ALIAS myanmartools)
elseif(TARGET myanmar-tools)
  add_library(myanmartools::myanmartools ALIAS myanmar-tools)
else()
  message(
    FATAL_ERROR
      "Myanmar-tools library target not found. Please check the library's CMakeLists.txt"
  )
endif()
