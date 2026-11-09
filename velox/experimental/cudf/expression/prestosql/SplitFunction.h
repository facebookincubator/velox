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

#include <string>

namespace facebook::velox::cudf_velox::prestosql {

/// Presto split with a non-empty constant delimiter and an optional constant
/// limit in [2, INT32_MAX]. Spark's regex split is not supported.
class SplitFunction : public CudfFunction {
 public:
  /// Checks a split call that matched the registered signature.
  static bool canEvaluate(const core::TypedExprPtr& expr);

  SplitFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  std::string delimiter_;
  // cuDF reads -1 as no limit.
  cudf::size_type maxSplitCount_{-1};
};

} // namespace facebook::velox::cudf_velox::prestosql
