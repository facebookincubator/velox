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

namespace facebook::velox::cudf_velox::prestosql {

/// Presto to_unixtime(timestamp) -> double.
/// Returns seconds since epoch as a double. cuDF TIMESTAMP_MICROSECONDS
/// are internally int64 us since epoch, so we reinterpret the underlying
/// data as INT64 (zero-copy) and divide by 1e6 in a single binary operation.
class ToUnixtimeFunction : public CudfFunction {
 public:
  /// Rejects a constant timestamp argument so a to_unixtime call that
  /// somehow reaches cuDF function selection without being constant-folded
  /// falls back to CPU cleanly, instead of eval() indexing into an empty
  /// inputColumns. See DateAddFunction::canEvaluate / DateTruncFunction::
  /// canEvaluate for the same defensive pattern.
  static bool canEvaluate(const std::shared_ptr<velox::exec::Expr>& expr);

  explicit ToUnixtimeFunction(const std::shared_ptr<velox::exec::Expr>& expr);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override;
};

} // namespace facebook::velox::cudf_velox::prestosql
