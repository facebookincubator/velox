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

if(DEFINED ENV{VELOX_PCRE2_URL})
  set(VELOX_PCRE2_SOURCE_URL "$ENV{VELOX_PCRE2_URL}")
else()
  set(VELOX_PCRE2_VERSION 10.47)
  set(
    VELOX_PCRE2_SOURCE_URL
    "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-${VELOX_PCRE2_VERSION}/pcre2-${VELOX_PCRE2_VERSION}.tar.gz"
  )
  set(
    VELOX_PCRE2_BUILD_SHA256_CHECKSUM
    c08ae2388ef333e8403e670ad70c0a11f1eed021fd88308d7e02f596fcd9dc16
  )
endif()

message(STATUS "Building PCRE2 ${VELOX_PCRE2_VERSION} from source")
FetchContent_Declare(
  pcre2
  URL ${VELOX_PCRE2_SOURCE_URL}
  URL_HASH SHA256=${VELOX_PCRE2_BUILD_SHA256_CHECKSUM}
)

set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_JIT ON CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_UNICODE ON CACHE BOOL "" FORCE)
set(PCRE2_STATIC_PIC ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(pcre2)

# Normalise the target name so consumers always link `pcre2-8::pcre2-8`.
if(TARGET pcre2-8-static AND NOT TARGET pcre2-8::pcre2-8)
  add_library(pcre2-8::pcre2-8 ALIAS pcre2-8-static)
elseif(TARGET pcre2-8 AND NOT TARGET pcre2-8::pcre2-8)
  add_library(pcre2-8::pcre2-8 ALIAS pcre2-8)
endif()

unset(BUILD_TESTING CACHE)
