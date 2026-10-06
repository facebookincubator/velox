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

#include "velox/common/base/Exceptions.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <optional>

namespace facebook::velox::cudf_velox::gpu_sfi {

/// The host-side mirror of the device GpuErrorKind, in its own header so
/// operators and the evaluator need not include a .cuh.
enum class ErrorClass : uint8_t {
  kNone = 0,
  /// A TRY above the expression may turn this row into a null.
  kUserError = 1,
  /// A TRY must not swallow this; the query has to fail.
  kRuntimeError = 2,
};

/// Collects the rows GPU simple-function launches declined during one
/// evaluation of one expression tree. A caller passes one only if it can act on
/// a declined row, by re-evaluating on the CPU so that Velox raises the error;
/// without one, a launch keeps its results. Only the owner of the whole tree
/// can decide, since a conditional above a node may discard exactly the rows
/// that node declined.
class GpuSfiErrors {
 public:
  GpuSfiErrors(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
      : stream_(stream), mr_(mr) {}

  /// The buffer a launch records into: one byte per row, holding an
  /// ErrorClass, shared by every launch in this evaluation so that it covers
  /// the whole tree. A launch writes only nonzero bytes, so a later launch
  /// cannot clear an earlier mark; when two decline a row, the last one wins.
  uint8_t* declinedRows(cudf::size_type numRows) {
    if (!buffer_.has_value()) {
      buffer_.emplace(numRows, stream_, mr_);
      // device_uvector is uninitialized, and a nonzero byte declines its row.
      CUDF_CUDA_TRY(
          cudaMemsetAsync(buffer_->data(), 0, buffer_->size(), stream_.get()));
    }
    VELOX_CHECK_LE(
        static_cast<std::size_t>(numRows),
        buffer_->size(),
        "A launch under one evaluation has more rows than the first");
    return buffer_->data();
  }

  /// Reduces the recorded bytes to the worst class in the batch. Called once by
  /// the owner after every launch is queued, since reading a device scalar
  /// synchronizes the stream. The classes are ordered, so the maximum answers
  /// both whether a row was declined and whether a TRY may swallow it.
  ErrorClass resolve() {
    if (!buffer_.has_value()) {
      return ErrorClass::kNone;
    }
    auto worst = cudf::reduce(
        cudf::column_view{
            cudf::data_type{cudf::type_id::UINT8},
            static_cast<cudf::size_type>(buffer_->size()),
            buffer_->data(),
            nullptr,
            0},
        *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type{cudf::type_id::UINT8},
        stream_,
        mr_);
    auto const* scalar =
        static_cast<cudf::numeric_scalar<uint8_t>*>(worst.get());
    if (!scalar->is_valid(stream_)) {
      return ErrorClass::kNone;
    }
    return static_cast<ErrorClass>(scalar->value(stream_));
  }

 private:
  cuda::stream_ref stream_;
  rmm::device_async_resource_ref mr_;
  std::optional<rmm::device_uvector<uint8_t>> buffer_;
};

} // namespace facebook::velox::cudf_velox::gpu_sfi
