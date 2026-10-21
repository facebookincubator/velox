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
#include <cudf/types.hpp>

#include <array>
#include <memory>

namespace facebook::velox::cudf_velox::prestosql {

/// from_unixtime(double) -> TIMESTAMP with the semantics of the Velox CPU
/// function, whose result is a millisecond timestamp: the whole seconds are
/// the floor of the input, the fraction rounds to the nearest millisecond
/// with halves away from zero and rolls a full second into the next one, NaN
/// gives the epoch, and an input at or past Timestamp::kMaxSeconds or
/// kMinSeconds, infinity included, takes the extreme millisecond timestamp.
/// The result then takes the configured cuDF timestamp unit: a millisecond
/// the unit cannot hold is pinned to the last millisecond it can, and the
/// seconds unit keeps the floor of the seconds, as a Velox timestamp converts
/// to it. The forms that return TIMESTAMP WITH TIME ZONE stay on the CPU.
class FromUnixtimeFunction : public CudfFunction {
 public:
  /// Returns true if expr is from_unixtime over a non-constant DOUBLE with a
  /// TIMESTAMP result.
  static bool canEvaluate(const core::TypedExprPtr& expr);

  /// Builds the function for expr, which must satisfy canEvaluate().
  FromUnixtimeFunction(
      const core::TypedExprPtr& expr,
      memory::MemoryPool* pool);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  // The configured cuDF timestamp type of the result.
  cudf::data_type timestampType_;
  // Ticks of timestampType_ in one second.
  std::unique_ptr<cudf::numeric_scalar<int64_t>> ticksPerSecond_;
};

/// to_unixtime(TIMESTAMP) -> DOUBLE with the semantics of the Velox CPU
/// function: the whole seconds, which are the floor of the timestamp, plus
/// the nanoseconds of the sub-second part over a billion, added in that
/// order so the doubles agree to the last bit. The form that takes TIMESTAMP
/// WITH TIME ZONE stays on the CPU.
class ToUnixtimeFunction : public CudfFunction {
 public:
  /// Returns true if expr is to_unixtime over a non-constant TIMESTAMP with a
  /// DOUBLE result.
  static bool canEvaluate(const core::TypedExprPtr& expr);

  /// Builds the function for expr, which must satisfy canEvaluate().
  ToUnixtimeFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  // Ticks of one second for each cuDF timestamp unit a column can carry,
  // built once so that eval() does not create and synchronize a scalar per
  // batch.
  std::array<std::unique_ptr<cudf::numeric_scalar<int64_t>>, 4> ticksPerSecond_;
};

} // namespace facebook::velox::cudf_velox::prestosql
