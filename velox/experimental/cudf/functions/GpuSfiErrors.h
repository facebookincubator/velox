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

#include <rmm/device_scalar.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

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

/// Collects whether GPU simple-function launches declined a row during one
/// evaluation of one expression tree, and the worst class they hit. A caller
/// passes one only if it can act on a declined row, by re-evaluating the batch
/// on the CPU so that Velox raises the error; without one, a launch keeps its
/// results. Only the owner of the whole tree can decide, since a conditional
/// above a node may discard exactly the rows that node declined.
///
/// One device word covers the tree: the owner re-evaluates the whole batch on
/// the CPU, so which row was declined is never read on the host.
class GpuSfiErrors {
 public:
  GpuSfiErrors(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
      : stream_(stream), mr_(mr) {}

  /// The word a launch records into, shared by every launch in this
  /// evaluation. A launch raises it with atomicMax, so a later launch cannot
  /// lower what an earlier one recorded.
  int32_t* worstKind() {
    if (!worstKind_.has_value()) {
      // Zero is kNone. A memset, where setting a value would copy it from
      // pageable host memory.
      worstKind_.emplace(stream_, mr_);
      worstKind_->set_value_to_zero_async(stream_);
    }
    return worstKind_->data();
  }

  /// Reads the word back. Called once by the owner after every launch is
  /// queued, since reading device memory synchronizes the stream. The classes
  /// are ordered, so the maximum answers both whether a row was declined and
  /// whether a TRY may swallow it.
  ErrorClass resolve() {
    if (!worstKind_.has_value()) {
      return ErrorClass::kNone;
    }
    return static_cast<ErrorClass>(worstKind_->value(stream_));
  }

 private:
  cuda::stream_ref stream_;
  rmm::device_async_resource_ref mr_;
  std::optional<rmm::device_scalar<int32_t>> worstKind_;
};

} // namespace facebook::velox::cudf_velox::gpu_sfi
