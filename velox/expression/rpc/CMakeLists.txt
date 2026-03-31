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

velox_install_library_headers()

velox_add_library(velox_async_rpc_function INTERFACE AsyncRPCFunction.h)

velox_link_libraries(
  velox_async_rpc_function
  INTERFACE velox_rpc_types velox_core velox_type velox_vector Folly::folly
)

velox_add_library(velox_rpc_function_stubs RPCFunctionStubs.cpp HEADERS RPCFunctionStubs.h)

velox_link_libraries(
  velox_rpc_function_stubs
  velox_expression_functions
  velox_expression
  glog::glog
)

velox_add_library(
  velox_async_rpc_function_registry
  AsyncRPCFunctionRegistry.cpp
  HEADERS
  AsyncRPCFunctionRegistry.h
)

velox_link_libraries(
  velox_async_rpc_function_registry
  velox_async_rpc_function
  velox_rpc_function_stubs
  velox_common_base
  velox_exception
  velox_expression_functions
  glog::glog
)
