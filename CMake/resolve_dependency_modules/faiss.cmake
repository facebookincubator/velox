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

set(VELOX_FAISS_BUILD_VERSION 1.12.0)
set(
  VELOX_FAISS_BUILD_SHA256_CHECKSUM
  561376d1a44771bf1230fabeef9c81643468009b45a585382cf38d3a7a94990a
)
set(
  VELOX_FAISS_SOURCE_URL
  "https://github.com/facebookresearch/faiss/archive/refs/tags/v${VELOX_FAISS_BUILD_VERSION}.tar.gz"
)

velox_resolve_dependency_url(FAISS)

FetchContent_Declare(
  faiss
  URL ${VELOX_FAISS_SOURCE_URL}
  URL_HASH ${VELOX_FAISS_BUILD_SHA256_CHECKSUM}
  SYSTEM
  EXCLUDE_FROM_ALL
)

# Set build options
block()
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTING OFF)
  set(CMAKE_BUILD_TYPE Release)
  set(FAISS_ENABLE_GPU OFF)
  set(FAISS_ENABLE_PYTHON OFF)
  set(FAISS_ENABLE_GPU_TESTS OFF)
  # Make FAISS available
  FetchContent_MakeAvailable(faiss)
  add_library(FAISS::faiss ALIAS faiss)
  unset(BUILD_TESTING CACHE)
  unset(BUILD_SHARED_LIBS CACHE)
endblock()
