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

# This creates a separate scope so any changed variables don't affect
# the rest of the build.
block()
  set(VELOX_LIBXML2_BUILD_VERSION 2.13.5)
  set(
    VELOX_LIBXML2_BUILD_SHA256_CHECKSUM
    74fc163217a3964257d3be39af943e08861263c4231f9ef5b496b6f6d4c7b2b6
  )
  string(
    CONCAT
    VELOX_LIBXML2_SOURCE_URL
    "https://download.gnome.org/sources/libxml2/2.13/"
    "libxml2-${VELOX_LIBXML2_BUILD_VERSION}.tar.xz"
  )

  velox_resolve_dependency_url(LIBXML2)

  message(STATUS "Building libxml2 from source")

  # Build a minimal, self-contained libxml2: only the parser and XPath engine
  # used by the Spark XML functions are needed. Optional subsystems and
  # external library integrations (Python bindings, programs, tests, iconv,
  # ICU, lzma, zlib, dynamic modules) are disabled so the bundled build does
  # not pull in additional system dependencies. Thread support stays ON because
  # the XPath engine relies on libxml2's per-thread error-handler state.
  set(LIBXML2_WITH_PYTHON OFF)
  set(LIBXML2_WITH_PROGRAMS OFF)
  set(LIBXML2_WITH_TESTS OFF)
  set(LIBXML2_WITH_ICONV OFF)
  set(LIBXML2_WITH_ICU OFF)
  set(LIBXML2_WITH_LZMA OFF)
  set(LIBXML2_WITH_ZLIB OFF)
  set(LIBXML2_WITH_HTTP OFF)
  set(LIBXML2_WITH_MODULES OFF)
  set(LIBXML2_WITH_THREADS ON)
  set(LIBXML2_WITH_XPATH ON)

  FetchContent_Declare(
    libxml2
    URL ${VELOX_LIBXML2_SOURCE_URL}
    URL_HASH ${VELOX_LIBXML2_BUILD_SHA256_CHECKSUM}
    OVERRIDE_FIND_PACKAGE
    SYSTEM
    EXCLUDE_FROM_ALL
  )

  set(BUILD_SHARED_LIBS OFF)
  set(CMAKE_BUILD_TYPE Release)

  FetchContent_MakeAvailable(libxml2)
endblock()
