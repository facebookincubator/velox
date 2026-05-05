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
#include "velox/functions/lib/DateTimeFormatter.h"

#include <cudf/scalar/scalar.hpp>

namespace facebook::velox::cudf_velox::prestosql {

/// date_add(unit, value, date) -> DATE.
/// Adds value units to date. unit must be a constant string from
/// {day, week, month, quarter, year}. value (bigint) and date may each be
/// a constant or a column, but at least one of the two must be a column.
/// Day and week units are handled with a duration_D add; month, quarter, and
/// year units are handled with cudf::datetime::add_calendrical_months.
class DateAddFunction : public CudfFunction {
 public:
  /// Returns true if expr matches the supported date_add shape: 3 inputs,
  /// DATE return type, DATE third argument, a constant unit string in the
  /// supported set, and not both value and date as constants.
  static bool canEvaluate(const std::shared_ptr<velox::exec::Expr>& expr);

  explicit DateAddFunction(const std::shared_ptr<velox::exec::Expr>& expr);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  /// Adds value*scale days (where scale comes from unit_) to dateCol.
  ColumnOrView evalDayBased(
      cudf::column_view dateCol,
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  /// Adds value*scale months (where scale comes from unit_) to dateCol.
  ColumnOrView evalMonthBased(
      cudf::column_view dateCol,
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Unit of the increment; one of {kDay, kWeek, kMonth, kQuarter, kYear}.
  functions::DateTimeUnit unit_{};
  // True if the value (second) argument is a constant ConstantExpr.
  bool valueIsLiteral_{};
  // True if the date (third) argument is a constant ConstantExpr.
  bool dateIsLiteral_{};
  // Position of the value column in inputColumns; meaningful only when
  // valueIsLiteral_ is false.
  size_t valueIdx_{};
  // Position of the date column in inputColumns; meaningful only when
  // dateIsLiteral_ is false.
  size_t dateIdx_{};
  // True if literalValue_ is non-null. When false, literalValue_ is unused
  // and the resulting cudf scalar carries a null validity bit.
  bool literalValueIsValid_{};
  // Constant value when valueIsLiteral_ is true.
  int64_t literalValue_{};
  // Pre-built scalar of the constant date input, when dateIsLiteral_ is true.
  std::unique_ptr<cudf::scalar> literalDate_;
};

} // namespace facebook::velox::cudf_velox::prestosql
