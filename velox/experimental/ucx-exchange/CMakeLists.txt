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

add_library(velox_ucx_exchange OBJECT ${UCX_EXCHANGE_SOURCES})

target_include_directories(velox_ucx_exchange PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

find_package(ucx REQUIRED)

target_link_libraries(
  velox_ucx_exchange
  PUBLIC ucxx::ucxx ucx::ucp
  PRIVATE cudf::cudf CUDA::cudart CUDA::cuda_driver Folly::folly velox_core ucx::ucs
)

if(VELOX_BUILD_TESTING)
  add_subdirectory(tests)
endif()
