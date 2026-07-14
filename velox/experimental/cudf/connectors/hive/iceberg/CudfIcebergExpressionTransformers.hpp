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

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/expressions.hpp>

#include <functional>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Builds a conservative filter over non-injected columns.
class CudfIcebergExpressionTransformer
    : private cudf::ast::detail::expression_transformer {
 public:
  CudfIcebergExpressionTransformer(
      cudf::ast::expression const& expression,
      std::vector<cudf::size_type> injectedColumnIndices);

  /// Returns the transformed filter, or nullptr if it is always true.
  cudf::ast::expression const* expression() const {
    return result_.expression;
  }

  /// Returns whether the input filter references an injected column.
  bool referencesInjectedColumn() const {
    return referencesInjectedColumn_;
  }

  /// Returns whether the filter or its column indices changed.
  bool changed() const {
    return changed_;
  }

 private:
  struct Result {
    cudf::ast::expression const* expression;
    bool wasRelaxed;
  };

  Result transform(cudf::ast::expression const& expression);

  std::reference_wrapper<cudf::ast::expression const> visit(
      cudf::ast::literal const& expression) override;

  std::reference_wrapper<cudf::ast::expression const> visit(
      cudf::ast::column_reference const& expression) override;

  std::reference_wrapper<cudf::ast::expression const> visit(
      cudf::ast::operation const& expression) override;

  std::reference_wrapper<cudf::ast::expression const> visit(
      cudf::ast::column_name_reference const& expression) override;

  std::vector<cudf::size_type> injectedColumnIndices_;
  cudf::ast::tree transformedTree_;
  Result result_{nullptr, false};
  bool referencesInjectedColumn_{false};
  bool changed_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
