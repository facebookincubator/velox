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

# 3.30.4 is the minimum version required by cudf
cmake_minimum_required(VERSION 3.30.4)

# rapids_cmake commit fa303cb from 2026-03-24
set(VELOX_rapids_cmake_VERSION 26.06)
set(VELOX_rapids_cmake_COMMIT fa303cb883f0e127fb2bb950d303626239050964)
set(
  VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM
  633616ce36fa21097483e793caa0dd94b355ea3735b6cb2a83e6f0fc10866bbd
)
set(
  VELOX_rapids_cmake_SOURCE_URL
  "https://github.com/rapidsai/rapids-cmake/archive/${VELOX_rapids_cmake_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(rapids_cmake)

# rmm commit ad99c11 from 2026-03-23
set(VELOX_rmm_VERSION 26.06)
set(VELOX_rmm_COMMIT ad99c114b62b9e1c8277563fe353ffb80589c84b)
set(
  VELOX_rmm_BUILD_SHA256_CHECKSUM
  baf203f4579bd778118360839bad57836aae4b07e482bec486ce5a850d92199d
)
set(VELOX_rmm_SOURCE_URL "https://github.com/rapidsai/rmm/archive/${VELOX_rmm_COMMIT}.tar.gz")
velox_resolve_dependency_url(rmm)

# kvikio commit b2bbfcc from 2026-03-24
set(VELOX_kvikio_VERSION 26.06)
set(VELOX_kvikio_COMMIT b2bbfcc3147fbadcdaf0e3f4b9737d9dd4bf76a0)
set(
  VELOX_kvikio_BUILD_SHA256_CHECKSUM
  d805843c9534a29815a66a1f047d4cac17cc6654da324a1f2a615330a8106ca1
)
set(
  VELOX_kvikio_SOURCE_URL
  "https://github.com/rapidsai/kvikio/archive/${VELOX_kvikio_COMMIT}.tar.gz"
)
velox_resolve_dependency_url(kvikio)

# cudf commit b593be9 from 2026-03-24
set(VELOX_cudf_VERSION 26.06 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT b593be9ab0bf144997efce09aaf9946f05113a39)
set(
  VELOX_cudf_BUILD_SHA256_CHECKSUM
  8f42f98a160388f45384f4ffa5f7c565c0532e6294dea1491b875cdfd28a70ec
)
set(VELOX_cudf_SOURCE_URL "https://github.com/rapidsai/cudf/archive/${VELOX_cudf_COMMIT}.tar.gz")
velox_resolve_dependency_url(cudf)

# Use block so we don't leak variables
block(SCOPE_FOR VARIABLES)
  # Setup libcudf build to not have testing components
  set(BUILD_TESTS OFF)
  set(CUDF_BUILD_TESTUTIL OFF)
  set(BUILD_SHARED_LIBS ON)

  FetchContent_Declare(
    rapids-cmake
    URL ${VELOX_rapids_cmake_SOURCE_URL}
    URL_HASH ${VELOX_rapids_cmake_BUILD_SHA256_CHECKSUM}
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    rmm
    URL ${VELOX_rmm_SOURCE_URL}
    URL_HASH ${VELOX_rmm_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    kvikio
    URL ${VELOX_kvikio_SOURCE_URL}
    URL_HASH ${VELOX_kvikio_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_Declare(
    cudf
    URL ${VELOX_cudf_SOURCE_URL}
    URL_HASH ${VELOX_cudf_BUILD_SHA256_CHECKSUM}
    SOURCE_SUBDIR
    cpp
    UPDATE_DISCONNECTED 1
  )

  FetchContent_MakeAvailable(cudf)

  # Host C++ in RMM/KvikIO/jitify includes CUDA headers; ensure Conda targets/*
  # include dirs are on the compile line even if a nested find_package skipped them.
  set(_velox_cuda_host_includes)
  if(DEFINED CUDAToolkit_INCLUDE_DIRS)
    list(APPEND _velox_cuda_host_includes ${CUDAToolkit_INCLUDE_DIRS})
  endif()
  if(DEFINED ENV{CONDA_PREFIX})
    file(GLOB _velox_cudf_cuda_targets "$ENV{CONDA_PREFIX}/targets/*")
    foreach(_d IN LISTS _velox_cudf_cuda_targets)
      if(EXISTS "${_d}/include/cuda.h")
        list(APPEND _velox_cuda_host_includes "${_d}/include")
        break()
      endif()
    endforeach()
  endif()
  if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/include/cuda.h")
    list(APPEND _velox_cuda_host_includes "$ENV{CUDA_PATH}/include")
  endif()
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include/cufile.h")
    list(APPEND _velox_cuda_host_includes "$ENV{CONDA_PREFIX}/include")
  endif()

  # KvikIO/RMM may already register Conda $PREFIX/include; appending our dirs then
  # loses to that path for <cufile.h>. Prepend (BEFORE) so toolkit/targets win when
  # they ship a complete cufile.h. Also detect a *truncated* cufile.h on the first
  # directory (in this order) that actually provides the file — a compile-only probe
  # can succeed against the toolkit while the real build picks Conda first.
  FetchContent_GetProperties(kvikio SOURCE_DIR _velox_kv_src)
  if(NOT _velox_kv_src AND EXISTS "${CMAKE_BINARY_DIR}/_deps/kvikio-src")
    set(_velox_kv_src "${CMAKE_BINARY_DIR}/_deps/kvikio-src")
  endif()

  set(_velox_need_cufile_batch_compat FALSE)
  if(DEFINED ENV{VELOX_FORCE_KVIKIO_CUFILE_COMPAT})
    set(_velox_need_cufile_batch_compat TRUE)
  elseif(_velox_cuda_host_includes)
    foreach(_d IN LISTS _velox_cuda_host_includes)
      if(EXISTS "${_d}/cufile.h")
        file(READ "${_d}/cufile.h" _velox_cufile_h_probe LIMIT 524288)
        if(NOT _velox_cufile_h_probe MATCHES "cuFileBatchIOSetUp")
          set(_velox_need_cufile_batch_compat TRUE)
        endif()
        break()
      endif()
    endforeach()
  endif()

  if(_velox_kv_src AND _velox_need_cufile_batch_compat)
    set(_velox_kvk_shim "${_velox_kv_src}/cpp/include/kvikio/shim")
    configure_file(
      "${CMAKE_SOURCE_DIR}/CMake/resolve_dependency_modules/cuda/velox_cufile_batch_stream_compat.h"
      "${_velox_kvk_shim}/velox_cufile_batch_stream_compat.h"
      COPYONLY
    )
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}"
        -D
        "KVIKIO_CUFILE_H_WRAPPER=${_velox_kvk_shim}/cufile_h_wrapper.hpp"
        -P
        "${CMAKE_SOURCE_DIR}/CMake/resolve_dependency_modules/cuda/patch_kvikio_cufile_h_wrapper_velox.cmake"
      RESULT_VARIABLE _velox_kv_patch_rv
    )
    if(NOT _velox_kv_patch_rv EQUAL 0)
      message(
        FATAL_ERROR
        "Velox: failed to patch KvikIO cufile_h_wrapper.hpp (cmake -P exit ${_velox_kv_patch_rv})"
      )
    endif()
    message(
      STATUS
      "Velox: KvikIO cufile.h lacks batch/stream API declarations; injected velox_cufile_batch_stream_compat.h"
    )
  endif()
  # target_* API cannot be used on ALIAS targets; map names to the underlying target.
  set(_velox_cuda_real_tgts)
  foreach(_velox_cuda_name IN ITEMS rmm::rmm rmm kvikio::kvikio kvikio jitify_preprocess)
    if(NOT TARGET ${_velox_cuda_name})
      continue()
    endif()
    get_target_property(_velox_cuda_aliased ${_velox_cuda_name} ALIASED_TARGET)
    if(_velox_cuda_aliased)
      set(_velox_cuda_effective ${_velox_cuda_aliased})
    else()
      set(_velox_cuda_effective ${_velox_cuda_name})
    endif()
    if(_velox_cuda_effective IN_LIST _velox_cuda_real_tgts)
      continue()
    endif()
    list(APPEND _velox_cuda_real_tgts ${_velox_cuda_effective})
    if(_velox_cuda_host_includes)
      target_include_directories(
        ${_velox_cuda_effective}
        BEFORE
        PRIVATE
        ${_velox_cuda_host_includes}
      )
    endif()
  endforeach()

  # CCCL deprecates <cuda/stream_ref> as a hard error under -Werror when pulled
  # through libcudacxx; RMM still triggers that path on some toolkit versions.
  foreach(_velox_rmm_name IN ITEMS rmm::rmm rmm)
    if(NOT TARGET ${_velox_rmm_name})
      continue()
    endif()
    get_target_property(_velox_rmm_aliased ${_velox_rmm_name} ALIASED_TARGET)
    if(_velox_rmm_aliased)
      set(_velox_rmm_real ${_velox_rmm_aliased})
    else()
      set(_velox_rmm_real ${_velox_rmm_name})
    endif()
    target_compile_definitions(
      ${_velox_rmm_real}
      PRIVATE
      CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
    )
    break()
  endforeach()

  # cudf sets all warnings as errors, and therefore fails to compile with velox
  # expanded set of warnings. We selectively disable problematic warnings just for
  # cudf
  target_compile_options(
    cudf
    PRIVATE -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-deprecated-copy -Wno-restrict
  )

  unset(BUILD_SHARED_LIBS)
  unset(BUILD_TESTING CACHE)
endblock()
