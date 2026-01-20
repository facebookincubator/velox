

#pragma once

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include <cudf/ast/expressions.hpp>

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
      const std::vector<int>& nestedIndices,
      const std::shared_ptr<CudfExpression>& node = nullptr)
      : dependent_column_index(depIndex),
        ins_name(name),
        new_column_index(newIndex),
        nested_dependent_column_indices(nestedIndices),
        cudf_expression(node) {}
};

} // namespace facebook::velox::cudf_velox
