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

#include "velox/experimental/cudf/CudfNoDefaults.h"

#include "velox/common/base/Exceptions.h"
#include "velox/type/Type.h"

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Builds a constant cuDF scalar on the given stream.
using ScalarFactory =
    std::function<std::unique_ptr<cudf::scalar>(cuda::stream_ref)>;

/// One output column of a projection: either an input channel or a constant.
struct ProjectionColumn {
  /// Input channel, or kConstantChannel when the column is a constant.
  column_index_t channel{kConstantChannel};

  /// Builds the constant. Empty when the column is an input channel.
  ScalarFactory makeScalar;

  /// Creates a column taken from input channel 'channel'.
  static ProjectionColumn field(column_index_t channel) {
    return {channel, nullptr};
  }

  /// Creates a column filled with the scalar 'makeScalar' returns.
  static ProjectionColumn constant(ScalarFactory makeScalar) {
    return {kConstantChannel, std::move(makeScalar)};
  }
};

/// Turns one input table into one output table per projection. Shared by
/// operators that emit several projections of each input batch, such as Expand
/// and GroupId.
///
/// An input column is copied for every use except its last across all
/// projections, where it is moved into the output.
class CudfProjectionCycle {
 public:
  /// 'projections' lists the output columns of each projection.
  /// 'numInputColumns' is the width of every input table.
  CudfProjectionCycle(
      std::vector<std::vector<ProjectionColumn>> projections,
      size_t numInputColumns);

  /// Returns true while the current input has projections left to output.
  bool hasInput() const {
    return hasInput_;
  }

  /// Takes ownership of the next input batch. Requires hasInput() to be false.
  void setInput(
      std::unique_ptr<cudf::table> input,
      cudf::size_type numRows,
      cuda::stream_ref stream);

  /// Returns the next projection of the current input. Drops the input after
  /// the last projection. Requires hasInput().
  std::unique_ptr<cudf::table> next(rmm::device_async_resource_ref outputMr);

  /// Number of rows in the current input, and in every output derived from it.
  cudf::size_type numRows() const {
    return numRows_;
  }

  /// Stream of the current input.
  cuda::stream_ref stream() const {
    return stream_;
  }

  /// Releases the input and cached scalars.
  void clear();

 private:
  // Output columns of each projection.
  const std::vector<std::vector<ProjectionColumn>> projections_;

  // Width of every input table, used to validate each batch before any
  // projection indexes into it.
  const size_t numInputColumns_;

  // Whether each output column takes ownership of its input column, one list
  // per projection. True only for the last use of an input column across all
  // projections.
  std::vector<std::vector<bool>> moveInputColumns_;

  // Scalars built from the constants on the first batch and reused for every
  // later batch. Null for input channels.
  std::vector<std::vector<std::unique_ptr<cudf::scalar>>> scalars_;

  // Stream that 'scalars_' were built on.
  std::optional<cuda::stream_ref> scalarStream_;

  // Columns of the current input, released from its table.
  std::vector<std::unique_ptr<cudf::column>> inputColumns_;

  // Signals pending input separately from 'inputColumns_', which is empty for
  // a zero-column input.
  bool hasInput_{false};

  // Row count of the current input.
  cudf::size_type numRows_{0};

  // Stream of the current input, used for all its outputs.
  cuda::stream_ref stream_{cudaStream_t{cudaStreamDefault}};

  // Index of the projection 'next' outputs.
  size_t projectionIndex_{0};
};

} // namespace facebook::velox::cudf_velox
