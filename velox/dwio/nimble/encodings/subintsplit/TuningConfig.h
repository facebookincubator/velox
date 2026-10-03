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

namespace folly {
class Executor;
} // namespace folly

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

  /// Whether a fitted slope * row + base (a line frame) or a per-row step (a
  /// step frame) may be subtracted from every value before planning its
  /// sections (see RowFrame.h). Either frame is kept only where it encodes
  /// smaller than the plain values; a read pays one multiply-add per row.
  bool rowFrame{true};

  /// Test and ablation only: keeps a fitted row frame without comparing the
  /// residuals against the values, so its cost where the encoder would have
  /// declined it can be measured. Inert unless rowFrame is set, and where no
  /// frame fits. Never set in a writer.
  bool rowFrameForceApply{false};

  /// Reversible transform applied to the sections, as a TransformId. Zero,
  /// the default, applies none. The section named by keySection is always
  /// left untransformed, since a key-derived permutation is rebuilt from it
  /// at read time.
  uint8_t transform{0};

  /// Lets the encoder choose per section whether to apply the key-derived
  /// transform, pricing it against the untransformed encoding and keeping the
  /// smaller. Ignores transform and is exclusive with forceApply. Costs one
  /// trial encode per candidate key per section. Off by default.
  bool autoTransform{false};

  /// Section whose values order a key-derived permutation, and which is
  /// therefore stored unpermuted. 0xFF, the default, means the encoder tries
  /// every section and keeps the one that encodes smallest, as does a section
  /// past the last one the stream's split has.
  uint8_t keySection{0xFF};

  /// Test and ablation only: skips the cost comparison that keeps a
  /// transform only where it encodes smaller, and applies transform to every
  /// eligible section regardless. Requires transform to name a real
  /// transform. With keySection at 0xFF the key is still searched: each
  /// candidate key forces the transform on every other section, and the
  /// smallest of those forced attempts is kept. Never set in a writer.
  bool forceApply{false};

  /// Chooses split boundaries with the hybrid planner instead of trusting the
  /// split DP's argmin. The DP serves as a shortlister; a small set of
  /// candidate plans is re-priced with section selection's own estimators and
  /// the winner is refined locally. Costs roughly twice the planning time.
  bool hybridPlanner{false};

  /// The most encoded size, as a fraction, that decode weighting
  /// (selector.decodeWeighting) may give up against what size-only selection
  /// would have chosen. Enforced wherever a decode-weighted choice is made:
  /// the split planner, the hybrid planner and a section's encoding
  /// selection. Inert at the default decode weight of zero.
  double maxSizeRegression{0.05};

  /// Executor sections are encoded on concurrently. Null encodes them one
  /// after another on the calling thread.
  folly::Executor* sectionExecutor{nullptr};

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
