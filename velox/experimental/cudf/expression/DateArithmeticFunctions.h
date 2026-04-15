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

namespace facebook::velox::cudf_velox {

/// Spark date_add(date, integer) -> date.
/// Adds a constant number of days (as tinyint/smallint/integer) to a date.
class DateAddFunction : public CudfFunction {
 public:
  explicit DateAddFunction(const std::shared_ptr<velox::exec::Expr>& expr);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  std::unique_ptr<cudf::scalar> value_;
};

/// plus(DATE, INTERVAL DAY TO SECOND) -> DATE.
/// Used by TPC-DS and Presto for date + interval arithmetic.
/// Converts the interval from milliseconds to days and adds to the date.
class DatePlusIntervalFunction : public CudfFunction {
 public:
  explicit DatePlusIntervalFunction(
      const std::shared_ptr<velox::exec::Expr>& expr);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override;
};

} // namespace facebook::velox::cudf_velox
