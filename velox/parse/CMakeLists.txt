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
# Leaf library that owns the out-of-line definitions for the
# `velox::core` expression class hierarchy declared in Expressions.h
# (CallExpr, CastExpr, FieldAccessExpr, ConstantExpr, LambdaExpr,
# WindowCallExpr, ...). Lifted out of velox_parse_expression so that
# velox_expression — which uses dynamic_cast on these classes — can
# pick up the typeinfo without taking a dependency on
# velox_parse_expression itself, which would close a cycle
# (velox_parse_expression already depends on velox_expression).
velox_add_library(velox_core_expressions Expressions.cpp HEADERS Expressions.h IExpr.h)
velox_link_libraries(velox_core_expressions velox_common_base velox_type Folly::folly)

velox_add_library(
  velox_parse_expression
  ExprRewriter.cpp
  SqlExpressionsParser.cpp
  HEADERS
  ExprRewriter.h
  SqlExpressionsParser.h
  TypeResolver.h
)
velox_link_libraries(
  velox_parse_expression
  velox_core_expressions
  velox_type
  velox_parse_utils
  velox_expression
)

velox_add_library(
  velox_parse_parser
  ExpressionsParser.cpp
  QueryPlanner.cpp
  HEADERS
  DuckLogicalOperator.h
  ExpressionsParser.h
  QueryPlanner.h
)
velox_link_libraries(velox_parse_parser velox_parse_expression velox_type velox_duckdb_parser)

velox_add_library(velox_parse_utils TypeResolver.cpp HEADERS PlanNodeIdGenerator.h)
velox_link_libraries(
  velox_parse_utils
  velox_buffer
  velox_exec
  velox_type
  velox_vector
  velox_function_registry
)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

velox_add_library(velox_plan_node_id_generator INTERFACE HEADERS PlanNodeIdGenerator.h)
