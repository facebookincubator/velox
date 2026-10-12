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

  /// Controls candidate generation and split selection. Huffman and
  /// DeltaBlock are withdrawn: each would be priced into where boundaries
  /// fall while no section can be encoded as it, and each costs a pass over
  /// the sample per grid cell. They must stay in step with
  /// nestedEncodingReadFactors, which decides what a section may be.
  SelectorConfig selector{.allowHuffman = false, .allowDeltaBlock = false};

  /// Encodings the planner may cost a section against. Empty means every
  /// encoding. A restricted set only narrows what the planner considers; it
  /// does not change the format.
  AllowedEncodings allowedEncodings{};

  /// Bounds values combined per decode pass.
  ///
  /// A sweep over 20 data patterns found 512 through 4,096 elements
  /// throughput-equivalent. The larger value amortizes nested dispatch while
  /// retaining cache locality.
  uint32_t decodeChunkSize{4'096};

  /// Folds Constant sections into one pre-shifted word when a stream is
  /// opened, so the decode loop never materialises them. Decode only.
  bool foldConstantSections{true};

  /// Decodes a stream whose one remaining section holds each value verbatim
  /// straight into the caller's buffer, skipping the scratch copy and the
  /// mask-and-shift pass. Decode only.
  bool passThrough{true};

  /// Decodes a block at a time on the readWithVisitor slow path instead of
  /// one value per section per call. Decode only; applies to streams with no
  /// transform, row frame or delta.
  bool visitorBlockBuffer{true};
};

/// Defines the production tuning used by every normal SubIntSplit encode and
/// decode path. Benchmarks and focused tests may pass an alternate config
/// directly to SubIntSplitEncoding.
inline const TuningConfig kDefaultTuningConfig{};

} // namespace facebook::nimble::subintsplit
