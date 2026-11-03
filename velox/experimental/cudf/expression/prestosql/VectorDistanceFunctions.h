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
#include "velox/experimental/cudf/expression/prestosql/VectorDistanceKernels.h"

#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::prestosql {

/// cosine_similarity, l2_squared and dot_product over array(real) -> real and
/// array(double) -> double. A null array or null element gives null, an empty
/// array gives NaN, a zero norm makes cosine_similarity NaN, and arrays of
/// different sizes raise a user error. Either argument may be a constant
/// without null elements. The real overload matches the FAISS-backed CPU
/// function. The double overload accumulates in double; the CPU overload
/// casts each element to float first.
class VectorDistanceFunction : public CudfFunction {
 public:
  /// Returns true if expr, a call that matched the registered signature, has
  /// a non-constant argument and any constant argument is a non-null array
  /// without null elements.
  static bool canEvaluate(const core::TypedExprPtr& expr);

  VectorDistanceFunction(
      VectorDistanceKind kind,
      const core::TypedExprPtr& expr,
      memory::MemoryPool* pool);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  const VectorDistanceKind kind_;

  // The CPU functions word the size error differently.
  const std::string sizeMismatchMessage_;

  const bool doublePrecision_;

  // Set for constant arguments; the other argument arrives as a column.
  // Only the pair that matches the call's element type is populated.
  std::optional<std::vector<float>> leftFloat_;
  std::optional<std::vector<float>> rightFloat_;
  std::optional<std::vector<double>> leftDouble_;
  std::optional<std::vector<double>> rightDouble_;
};

/// Registers the GPU vector distance functions under the given prefix.
void registerVectorDistanceFunctions(const std::string& prefix);

} // namespace facebook::velox::cudf_velox::prestosql
