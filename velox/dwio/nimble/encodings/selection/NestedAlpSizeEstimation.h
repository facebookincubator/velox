/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"

namespace facebook::nimble::detail {

/// Shares bounded sampling and scalar child-size heuristics for ALP and ALPRD.
/// Uses the existing estimators' size convention for default-enabled codecs.
class NestedAlpSizeEstimation {
 public:
  /// Picks a deterministic offset within an evenly sized sampling interval.
  static uint32_t
  sampledRowIndex(uint32_t sampleIndex, uint32_t numSamples, uint32_t numRows);

  /// Estimates a non-empty integer child using its observed value range and
  /// target row count. Constant stores one value; other ranges use the smaller
  /// of FixedBitWidth and Trivial. Child writers select encodings
  /// independently.
  template <typename T>
  static uint64_t estimateChildSize(
      uint32_t numRows,
      uint64_t minValue,
      uint64_t maxValue,
      const Encoding::Options& options);
};

} // namespace facebook::nimble::detail
