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
set(
  UCX_EXCHANGE_SOURCES
  UcxPartitionedOutput.cpp
  UcxOutputQueueManager.cpp
  UcxQueues.cpp
  UcxExchange.cpp
  UcxExchangeProtocol.cpp
  UcxCompressionCostModel.cpp
  UcxExchangeSource.cpp
  UcxExchangeServer.cpp
  UcxExchangeQueue.cpp
  UcxExchangeClient.cpp
  Communicator.cpp
  Acceptor.cpp
  EndpointRef.cpp
  NetUtil.cpp
  IntraNodeTransferRegistry.cpp
)

# Generate a header containing the module basenames (without .cpp extension)
# for use by Communicator to set per-file VLOG levels.
set(UCX_EXCHANGE_MODULE_LIST "")
foreach(src ${UCX_EXCHANGE_SOURCES})
  string(REGEX REPLACE "\\.cpp$" "" module "${src}")
  string(APPEND UCX_EXCHANGE_MODULE_LIST "\"${module}\",\n    ")
endforeach()
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/UcxExchangeModules.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/UcxExchangeModules.h
  @ONLY
)

include(FetchContent)

# DietGPU supplies the byte-oriented rANS implementation. Pin the source and
# checksum so builds are reproducible without carrying a private source copy.
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
  UCX_EXCHANGE_CUDA_SOURCES
  UcxCompression.cu
  UcxColumnCodec.cu
  UcxFloat64Codec.cu
  UcxFloat64AlpCodec.cu
  ${DIETGPU_ROOT}/ans/GpuANSEncode.cu
  ${DIETGPU_ROOT}/ans/GpuANSDecode.cu
  ${DIETGPU_ROOT}/ans/GpuANSInfo.cu
  ${DIETGPU_ROOT}/utils/DeviceUtils.cpp
  ${DIETGPU_ROOT}/utils/StackDeviceMemory.cpp
)

# Third-party CUDA code is not held to Velox warning levels.
set_source_files_properties(
  ${DIETGPU_ROOT}/ans/GpuANSEncode.cu
  ${DIETGPU_ROOT}/ans/GpuANSDecode.cu
  ${DIETGPU_ROOT}/ans/GpuANSInfo.cu
  PROPERTIES COMPILE_OPTIONS "-w;--diag-suppress=68,186,554"
)

# DietGPU headers trigger these nvcc diagnostics in wrapper translation units.
set_source_files_properties(
  UcxCompression.cu
  UcxColumnCodec.cu
  UcxFloat64Codec.cu
  UcxFloat64AlpCodec.cu
  PROPERTIES COMPILE_OPTIONS "--diag-suppress=68,186,554"
)

add_library(velox_ucx_exchange OBJECT ${UCX_EXCHANGE_SOURCES} ${UCX_EXCHANGE_CUDA_SOURCES})

target_include_directories(
  velox_ucx_exchange
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${dietgpu_source_SOURCE_DIR}
)

find_package(ucx REQUIRED)

target_link_libraries(
  velox_ucx_exchange
  PUBLIC ucxx::ucxx ucx::ucp nvtx3::nvtx3-cpp
  PRIVATE cudf::cudf CUDA::cudart CUDA::cuda_driver Folly::folly velox_core ucx::ucs
)

if(VELOX_BUILD_TESTING)
  add_subdirectory(tests)
endif()
