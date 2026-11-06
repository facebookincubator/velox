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

/// array_join(array(varchar), separator) -> VARCHAR with Presto semantics:
/// the elements of each array joined by the constant separator. A null
/// element is skipped, an empty array gives an empty string and a null array
/// gives null. The form with a null replacement stays on the CPU, since cuDF
/// joins an all-null array to an empty string where Presto writes the
/// replacement; a separator column and arrays of other element types stay
/// there as well.
class ArrayJoinFunction : public CudfFunction {
 public:
  /// Returns true if expr, an array_join call that matched the registered
  /// signature, takes a non-constant array and a non-null constant separator.
  static bool canEvaluate(const core::TypedExprPtr& expr);

  ArrayJoinFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  // Placed between consecutive non-null elements.
  std::string separator_;
};

} // namespace facebook::velox::cudf_velox::prestosql
