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
  velox_parse_expression
  ExprRewriter.cpp
  Expressions.cpp
  SqlExpressionsParser.cpp
  HEADERS
  ExprRewriter.h
  Expressions.h
  IExpr.h
  SqlExpressionsParser.h
  TypeResolver.h
)
velox_link_libraries(velox_parse_expression velox_type velox_parse_utils velox_expression)

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

# QueryPlanner.cpp includes <duckdb.hpp> (via QueryPlanner.h /
# DuckLogicalOperator.h), whose vendored headers do not compile as C++23
# (constexpr ~unique_ptr over an incomplete duckdb::QueryNode). Compile it as
# C++20. velox_add_library may alias velox_parse_parser to the single velox
# mono-library defined in another directory, so the source property must be set
# in that target's directory scope (TARGET_DIRECTORY).
get_target_property(_velox_parse_parser_tgt velox_parse_parser ALIASED_TARGET)
if(NOT _velox_parse_parser_tgt)
  set(_velox_parse_parser_tgt velox_parse_parser)
endif()
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/QueryPlanner.cpp
  TARGET_DIRECTORY ${_velox_parse_parser_tgt}
  PROPERTIES COMPILE_OPTIONS -std=gnu++20
)

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
