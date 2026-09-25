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

#include <unordered_set>

#include "velox/dwio/nimble/encodings/common/EncodingType.h"

namespace facebook::nimble::subintsplit {

/// SubIntSplit planner and decoder settings, held by Encoding::Options as
/// Encoding::Options::subIntSplit.
struct Options {
  /// Encodings SubIntSplit may cost a section against when choosing
  /// splits. Empty, the default, means every encoding. A restricted set
  /// only narrows what the selector considers; it does not change the
  /// format.
  std::unordered_set<EncodingType> allowedEncodings;

  /// Whether SubIntSplit's split planner may cost a bit range as Huffman.
  /// False by default: no section encoding can actually select Huffman, so
  /// pricing it only steers split boundaries toward an encoding nothing
  /// will use. Set true to price it anyway.
  bool allowHuffman{false};

  /// Allows the SubIntSplit split planner to cost segments as DeltaBlock.
  /// False by default: DeltaBlock's serial prefix-sum decode does not
  /// vectorize and its per-block baselines pay off only for random access,
  /// not the contiguous scans this was measured on, while its cost model
  /// is expensive to evaluate per grid cell. Must stay in sync with
  /// whether DeltaBlock appears in the SubIntSplit nested candidate list
  /// in nestedEncodingReadFactors, since the planner and section selection
  /// need to agree on what is available.
  bool allowDeltaBlock{false};

  /// Trims the bit planes constant across SubIntSplit's planner sample
  /// before the split DP, storing each constant edge as one Constant section
  /// and scoring the grid over the varying planes alone. On by default, as
  /// the planner has always trimmed; turning it off changes the plan and so
  /// the encoded output. Ignored for an edge narrower than the minimum
  /// section width, or when allowedEncodings excludes Constant.
  bool trimConstantPlanes{true};

  /// Folds SubIntSplit's Constant sections into one pre-shifted word when a
  /// stream is opened, so the decode loop never materialises them. Decode
  /// only: encoded output is unchanged. Applies to streams without section
  /// transforms. On by default.
  bool foldConstantSections{true};

  /// Decodes a SubIntSplit stream whose one remaining section holds each
  /// value verbatim straight into the caller's buffer, skipping the
  /// scratch copy and the mask-and-shift pass. Decode only. On by default.
  bool passThrough{true};

  /// Decodes SubIntSplit a block at a time on the readWithVisitor slow
  /// path, instead of one value per section per call. Decode only;
  /// applies to streams with no transform, row frame or delta. On by
  /// default, as the slow path has always decoded in blocks.
  bool visitorBlockBuffer{true};

  /// Whether SubIntSplit may subtract a fitted slope * row + base (a line
  /// frame) or a per-row step (a step frame) from every value before
  /// planning its sections (see RowFrame.h). Either frame is kept only where
  /// it encodes smaller than the plain values; a read pays one multiply-add
  /// per row. On by default.
  bool rowFrame{true};

  /// Test and ablation only: keeps a fitted row frame without comparing the
  /// residuals against the values, so its cost where the encoder would have
  /// declined it can be measured. Inert unless rowFrame is set, and where no
  /// frame fits. Never set in a writer.
  bool rowFrameForceApply{false};
};

} // namespace facebook::nimble::subintsplit
