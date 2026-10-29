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

# cuVS must be resolved after cuDF so it reuses cuDF's rapids-cmake, rmm,
# kvikio, CCCL and cuco targets. Keep the commit on the same RAPIDS release as
# cudf.cmake.

# cuvs commit facd61d from 2026-10-15 (main branch)
set(VELOX_cuvs_VERSION 26.12)
set(VELOX_cuvs_COMMIT facd61db619c812aed2a3b3961f4bc50e38d8e58)
set(
  VELOX_cuvs_BUILD_SHA256_CHECKSUM
  e2c9f1dcd2724807205d43452a174300f51f3a3d1b109542e507d1ecca3f6570
)
set(VELOX_cuvs_SOURCE_URL "https://github.com/rapidsai/cuvs/archive/${VELOX_cuvs_COMMIT}.tar.gz")
velox_resolve_dependency_url(cuvs)

# cuVS clones RAFT at the v${VERSION} tag, which does not exist until the
# release is tagged.
# raft commit 2d86756 from 2026-10-15 (main branch)
set(VELOX_raft_COMMIT 2d867567c7f0c8dd6981be400487d600bbfc3dbf)
set(
  VELOX_raft_BUILD_SHA256_CHECKSUM
  5994032f35cd486b0dca261931ad821ddc7f88f06a17e22abb83b596a3116b90
)
set(VELOX_raft_SOURCE_URL "https://github.com/NVIDIA/raft/archive/${VELOX_raft_COMMIT}.tar.gz")
velox_resolve_dependency_url(raft)

if(NOT TARGET rmm::rmm)
  message(FATAL_ERROR "cuVS must be resolved after cuDF.")
endif()

block(SCOPE_FOR VARIABLES)
  set(BUILD_SHARED_LIBS ON)
  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY OFF)
  set(BUILD_CUVS_BENCH OFF)
  set(BUILD_CAGRA_HNSWLIB OFF)
  set(BUILD_MG_ALGOS OFF)
  set(CUVS_COMPILE_DYNAMIC_ONLY ON)
  set(CUVS_STATIC_RAPIDS_LIBRARIES OFF)
  set(DETECT_CONDA_ENV OFF)

  # Download only: cuVS adds RAFT itself through CPM_raft_SOURCE.
  FetchContent_Declare(
    velox_raft
    URL ${VELOX_raft_SOURCE_URL}
    URL_HASH ${VELOX_raft_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    no-cmake
    UPDATE_DISCONNECTED 1
  )
  FetchContent_MakeAvailable(velox_raft)
  set(CPM_raft_SOURCE ${velox_raft_SOURCE_DIR})

  # cuDF already added rtcx, so cuVS reuses the target and skips including the
  # module that defines generate_jit_lto_kernels().
  FetchContent_GetProperties(rtcx SOURCE_DIR VELOX_RTCX_SOURCE_DIR)
  if(VELOX_RTCX_SOURCE_DIR)
    include(${VELOX_RTCX_SOURCE_DIR}/cmake/modules/generate_jit_lto_kernels.cmake)
  endif()

  FetchContent_Declare(
    cuvs
    URL ${VELOX_cuvs_SOURCE_URL}
    URL_HASH ${VELOX_cuvs_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
    SYSTEM
  )
  FetchContent_MakeAvailable(cuvs)
  unset(BUILD_TESTING CACHE)
endblock()
