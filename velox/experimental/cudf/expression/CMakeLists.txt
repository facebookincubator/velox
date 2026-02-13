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

add_library(velox_cudf_expression_registry ExpressionEvaluatorRegistry.cpp)

target_link_libraries(
  velox_cudf_expression_registry
  PUBLIC cudf::cudf
  PRIVATE velox_exception velox_common_base
)

add_library(
  velox_cudf_expression
  AstExpression.cpp
  ExpressionEvaluator.cpp
  JitExpression.cpp
  SubfieldFiltersToAst.cpp
)

target_link_libraries(
  velox_cudf_expression
  PUBLIC cudf::cudf
  PRIVATE arrow velox_exception velox_common_base velox_cudf_vector velox_cudf_expression_registry
)

# Note: This may seem circular (exec -> expression -> exec) but it's resolved
# by the linker because we only use exec's compiled libraries here, not headers

target_compile_options(velox_cudf_expression PRIVATE -Wno-missing-field-initializers)
