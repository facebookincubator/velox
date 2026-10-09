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

#include "velox/expression/FunctionCallToSpecialForm.h"

#include <vector>

namespace facebook::velox::functions::sparksql {

/// Compiles Spark's format_string and printf calls into a conditional special
/// form that evaluates the format pattern before the remaining arguments.
///
/// Supports the following subset of Java's Formatter syntax:
///   - '%%' for a literal percent sign.
///   - Bare '%s' for VARCHAR, BOOLEAN, TINYINT, SMALLINT, INTEGER, and BIGINT.
///   - '%d' for integral types with an optional width and the '-', '+',
///     space, and '0' flags.
///   - '%o', '%x', and '%X' for integral types with an optional width and the
///     '-' and '0' flags.
/// Widths are limited to 1,048,576. Precision, argument indexes, other
/// conversions, and other flags raise a user error.
///
/// Follow-up work, including '%s' width and alignment, integral grouping, and
/// '%f' for REAL and DOUBLE, is tracked in
/// https://github.com/facebookincubator/velox/issues/19471.
class FormatStringCallToSpecialForm : public exec::FunctionCallToSpecialForm {
 public:
  /// Resolves the return type to VARCHAR.
  TypePtr resolveType(const std::vector<TypePtr>& argumentTypes) override;

  /// Constructs the conditional format-string expression.
  exec::ExprPtr constructSpecialForm(
      const TypePtr& type,
      std::vector<exec::ExprPtr>&& arguments,
      bool trackCpuUsage,
      const core::QueryConfig& config) override;

  /// Canonical Spark function name.
  static constexpr const char* kFormatString = "format_string";

  /// Spark alias for format_string.
  static constexpr const char* kPrintf = "printf";
};

} // namespace facebook::velox::functions::sparksql
