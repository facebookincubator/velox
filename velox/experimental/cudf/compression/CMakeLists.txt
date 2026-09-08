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

include(FetchContent)

# PR #18746's packed-column codec uses DietGPU byte-rANS after numeric
# transforms. Keep the pinned source and checksum shared by the cache and UCX
# exchange instead of making codec availability depend on UCX being installed.
FetchContent_Declare(
  dietgpu_source
  URL
    https://github.com/facebookresearch/dietgpu/archive/a4d70a14066d2c3e5fe1849b3723e4cd423eee7e.tar.gz
  URL_HASH SHA256=fd6f367e0d678df35f0c2c66d624ccab24ffef3cd3db19944893684f023ff6f2
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR
  velox_source_only
)
FetchContent_MakeAvailable(dietgpu_source)
set(DIETGPU_ROOT ${dietgpu_source_SOURCE_DIR}/dietgpu)
set(
  CUDF_PACKED_CODEC_SOURCES
  PackedUcxCompression.cu
  PackedUcxColumnCodec.cu
  PackedUcxFloat64Codec.cu
  PackedUcxFloat64AlpCodec.cu
  ${DIETGPU_ROOT}/ans/GpuANSEncode.cu
  ${DIETGPU_ROOT}/ans/GpuANSDecode.cu
  ${DIETGPU_ROOT}/ans/GpuANSInfo.cu
  ${DIETGPU_ROOT}/utils/DeviceUtils.cpp
  ${DIETGPU_ROOT}/utils/StackDeviceMemory.cpp
)

set_source_files_properties(
  ${DIETGPU_ROOT}/ans/GpuANSEncode.cu
  ${DIETGPU_ROOT}/ans/GpuANSDecode.cu
  ${DIETGPU_ROOT}/ans/GpuANSInfo.cu
  PROPERTIES COMPILE_OPTIONS "-w;--diag-suppress=68,186,554"
)

set_source_files_properties(
  PackedUcxCompression.cu
  PackedUcxColumnCodec.cu
  PackedUcxFloat64Codec.cu
  PackedUcxFloat64AlpCodec.cu
  PROPERTIES COMPILE_OPTIONS "--diag-suppress=68,186,554"
)

add_library(velox_cudf_packed_codec STATIC ${CUDF_PACKED_CODEC_SOURCES})

target_include_directories(
  velox_cudf_packed_codec
  PRIVATE ${dietgpu_source_SOURCE_DIR}
)

target_link_libraries(
  velox_cudf_packed_codec
  PUBLIC cudf::cudf
  PRIVATE CUDA::cudart CUDA::cuda_driver Folly::folly
)
