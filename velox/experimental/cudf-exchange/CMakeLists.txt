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
velox_add_library(
  velox_cudf_exchange
  CudfPartitionedOutput.cpp
  CudfOutputQueueManager.cpp
  CudfQueues.cpp
  HybridExchange.cpp
  ExchangeClientFacade.cpp
  CudfExchangeProtocol.cpp
  CudfExchangeSource.cpp
  CudfExchangeServer.cpp
  CudfExchangeQueue.cpp
  CudfExchangeClient.cpp
  Communicator.cpp
  Acceptor.cpp
  EndpointRef.cpp
  NetUtil.cpp
  IntraNodeTransferRegistry.cpp
)

find_package(ucx REQUIRED)

velox_link_libraries(
  velox_cudf_exchange
  PUBLIC ucxx::ucxx ucx::ucp ucx::ucs
  PRIVATE cudf::cudf CUDA::cudart CUDA::cuda_driver Folly::folly velox_core
)

if(VELOX_BUILD_TESTING)
  add_subdirectory(tests)
endif()
