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

#include <cstdint>

#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"

namespace facebook::nimble::subintsplit {

/// Groups SubIntSplit's algorithm-local planner and decoder settings.
struct TuningConfig {
  /// Controls deterministic sampling for split planning.
  SamplerConfig sampler{};

  /// Controls candidate generation and split selection.
  SelectorConfig selector{};

  /// Bounds values combined per decode pass.
  ///
  /// A sweep over 20 data patterns found 512 through 4,096 elements
  /// throughput-equivalent. The larger value amortizes nested dispatch while
  /// retaining cache locality.
  uint32_t decodeChunkSize{4'096};
};

/// Defines the production tuning used by every normal SubIntSplit encode and
/// decode path. Benchmarks and focused tests may pass an alternate config
/// directly to SubIntSplitEncoding.
inline constexpr TuningConfig kDefaultTuningConfig{};

} // namespace facebook::nimble::subintsplit
