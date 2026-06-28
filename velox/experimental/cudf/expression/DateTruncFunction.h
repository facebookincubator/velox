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

namespace facebook::velox::cudf_velox {

/// Truncates a timestamp or date to the specified unit (second, minute, hour,
/// day, week, month, quarter, year). Registered for both Spark (timestamp
/// only) and Presto (timestamp + date).
class DateTruncFunction : public CudfFunction {
 public:
  static bool canEvaluate(const std::shared_ptr<velox::exec::Expr>& expr);

  explicit DateTruncFunction(const std::shared_ptr<velox::exec::Expr>& expr);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  functions::DateTimeUnit unit_;
  bool isTimestamp_{false};
  bool isDate_{false};
};

} // namespace facebook::velox::cudf_velox
