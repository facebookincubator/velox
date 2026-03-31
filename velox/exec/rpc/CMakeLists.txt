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

# RPCNode is part of velox_core (in core/PlanNode.h)

velox_add_library(velox_rpc_state RPCState.cpp HEADERS RPCState.h)

velox_link_libraries(
  velox_rpc_state
  velox_rpc_types
  velox_common_base
  velox_future
  velox_vector
  Folly::folly
)

velox_add_library(velox_rpc_rate_limiter RPCRateLimiter.cpp HEADERS RPCRateLimiter.h)

velox_link_libraries(velox_rpc_rate_limiter velox_future)

velox_add_library(velox_rpc_operator RPCOperator.cpp HEADERS RPCOperator.h)

velox_link_libraries(
  velox_rpc_operator
  velox_rpc_state
  velox_rpc_rate_limiter
  velox_async_rpc_function
  velox_async_rpc_function_registry
  velox_exec
  velox_expression
  velox_vector
  velox_buffer
  velox_future
  Folly::folly
)

velox_add_library(
  velox_rpc_plan_node_translator
  RPCPlanNodeTranslator.cpp
  HEADERS
  RPCPlanNodeTranslator.h
)

velox_link_libraries(velox_rpc_plan_node_translator velox_rpc_operator velox_exec)

if(${VELOX_BUILD_TESTING})
  velox_add_library(velox_demo_rpc_function tests/DemoRPCFunction.cpp)

  velox_link_libraries(
    velox_demo_rpc_function
    velox_mock_rpc_client
    velox_expression_functions
    velox_async_rpc_function
  )

  add_subdirectory(tests)
endif()
