/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include <cudf/ast/expressions.hpp>

#include <utility>

namespace facebook::velox::cudf_velox {

// Pre-compute instructions for the expression,
// for ops that are not supported by cudf::ast
struct PrecomputeInstruction {
  int dependent_column_index;
  std::string ins_name;
  int new_column_index;
  std::vector<int> nested_dependent_column_indices;
  std::shared_ptr<CudfExpression> cudf_expression;

  // Constructor to initialize the struct with values
  PrecomputeInstruction(
      int depIndex,
      const std::string& name,
      int newIndex,
      const std::shared_ptr<CudfExpression>& node = nullptr)
      : dependent_column_index(depIndex),
        ins_name(name),
        new_column_index(newIndex),
        cudf_expression(node) {}

  // TODO (dm): This two ctor situation is crazy.
  PrecomputeInstruction(
      int depIndex,
      const std::string& name,
      int newIndex,
      std::vector<int>&& nestedIndices,
      const std::shared_ptr<CudfExpression>& node = nullptr)
      : dependent_column_index(depIndex),
        ins_name(name),
        new_column_index(newIndex),
        nested_dependent_column_indices(std::move(nestedIndices)),
        cudf_expression(node) {}
};

} // namespace facebook::velox::cudf_velox
