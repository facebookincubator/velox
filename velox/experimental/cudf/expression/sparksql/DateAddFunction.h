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

namespace facebook::velox::cudf_velox::sparksql {

/// Spark date_add(date, days) -> DATE.
/// Adds a constant number of days to a date column. days must be a constant
/// (tinyint, smallint, or integer); date must be a column.
/// Note: signature differs from Presto date_add, which takes (unit, value,
/// date).
class DateAddFunction : public CudfFunction {
 public:
  explicit DateAddFunction(const std::shared_ptr<velox::exec::Expr>& expr);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  // Pre-computed duration_D scalar holding the days argument.
  std::unique_ptr<cudf::scalar> value_;
};

} // namespace facebook::velox::cudf_velox::sparksql
