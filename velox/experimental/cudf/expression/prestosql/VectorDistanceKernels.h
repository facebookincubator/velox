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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <cuda/stream_ref>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <span>

namespace facebook::velox::cudf_velox::prestosql {

enum class VectorDistanceKind {
  kCosineSimilarity,
  kL2Squared,
  kDotProduct,
};

/// One argument of a vector distance function: either a list column of T or a
/// constant non-null vector without null elements.
template <typename T>
struct VectorDistanceOperand {
  std::optional<cudf::column_view> column;
  std::span<const T> constant;
};

struct VectorDistanceResult {
  std::unique_ptr<cudf::column> column;
  /// True when a row with two non-null, null-free arrays had different sizes.
  bool sizeMismatch{false};
};

/// Computes the distance between the arrays in each row as T. A row is null
/// when either array is null or holds a null element. Rows of size zero, and
/// cosine rows with a zero norm, are NaN. At least one operand must be a
/// column; two columns must have equal sizes. T is float or double.
template <typename T>
VectorDistanceResult vectorDistance(
    VectorDistanceKind kind,
    const VectorDistanceOperand<T>& left,
    const VectorDistanceOperand<T>& right,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

extern template VectorDistanceResult vectorDistance<float>(
    VectorDistanceKind kind,
    const VectorDistanceOperand<float>& left,
    const VectorDistanceOperand<float>& right,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

extern template VectorDistanceResult vectorDistance<double>(
    VectorDistanceKind kind,
    const VectorDistanceOperand<double>& left,
    const VectorDistanceOperand<double>& right,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::prestosql
