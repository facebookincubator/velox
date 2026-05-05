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

#include <cudf/scalar/scalar.hpp>

namespace facebook::velox::cudf_velox::prestosql {

/// plus(DATE, INTERVAL DAY TO SECOND) -> DATE.
/// Converts the interval from milliseconds to days and adds to the date.
/// Handles both constant and column interval inputs.
class DatePlusIntervalFunction : public CudfFunction {
 public:
  explicit DatePlusIntervalFunction(
      const std::shared_ptr<velox::exec::Expr>& expr);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  /// Pre-computed duration scalar for constant interval inputs.
  std::unique_ptr<cudf::scalar> durationDaysLiteral_;
};

} // namespace facebook::velox::cudf_velox::prestosql
