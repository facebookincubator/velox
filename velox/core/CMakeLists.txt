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
if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

velox_add_library(
  velox_core
  Expressions.cpp
  ITypedExpr.cpp
  PlanConsistencyChecker.cpp
  PlanFragment.cpp
  PlanNode.cpp
  QueryConfig.cpp
  QueryCtx.cpp
  SimpleFunctionMetadata.cpp
  TableWriteTraits.cpp
  HEADERS
  CoreTypeSystem.h
  Expressions.h
  ITypedExpr.h
  PlanConsistencyChecker.h
  PlanFragment.h
  PlanNode.h
  QueryConfig.h
  QueryCtx.h
  SimpleFunctionMetadata.h
  TableWriteTraits.h
)

velox_link_libraries(
  velox_core
  PUBLIC
    velox_arrow_bridge
    velox_caching
    velox_common_config
    velox_config_property
    velox_connector
    velox_exception
    velox_expression_functions
    velox_memory
    velox_type
    velox_vector
    Boost::headers
    Folly::folly
    fmt::fmt
  PRIVATE velox_encode
)

velox_add_library(velox_query_config_provider QueryConfigProvider.cpp HEADERS QueryConfigProvider.h)
velox_link_libraries(velox_query_config_provider PUBLIC velox_core velox_config_property)

velox_add_library(velox_metaprogramming INTERFACE HEADERS Metaprogramming.h)

velox_add_library(velox_expression_evaluator INTERFACE HEADERS ExpressionEvaluator.h)
