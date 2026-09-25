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
#include "velox/dwio/nimble/encodings/subintsplit/DecodeCost.h"

namespace folly {
class Executor;
} // namespace folly

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

  /// Executor SubIntSplit encodes its sections on concurrently. Null, the
  /// default, encodes them one after another on the calling thread.
  folly::Executor* sectionExecutor{nullptr};

  /// How much a section's decode cost counts against its encoded size when
  /// the split planner chooses boundaries and encodings. Zero, the default,
  /// is size-only selection. The unit is bytes of encoded size per
  /// nanosecond per row of decode; a caller picks this as an exchange rate,
  /// not a measurement. See DecodeCost.h for where the per-encoding rates
  /// come from.
  double decodeWeight{0.0};

  /// The read shape section decode is costed for when decodeWeight is
  /// non-zero. Encodings do not rank the same way on every access pattern,
  /// so weighting decode without naming the pattern would optimise for
  /// whichever one the rates were fitted on.
  DecodeAccessPattern decodeAccessPattern{DecodeAccessPattern::Bulk};

  /// The reader section decode is costed for when decodeWeight is non-zero.
  /// Pricing section opens for a reader that amortises them buys faster
  /// opens at the cost of slower reads.
  DecodeReadPath decodeReadPath{DecodeReadPath::Cursor};

  /// Whether these options are the ones a SubIntSplit section is being
  /// encoded with, rather than a column's own options. Set only by
  /// sectionEncodingOptions and read by encoding selection to decide whether
  /// decodeWeight applies. Marks the whole subtree below a section, so a
  /// nested stream is priced on decode too.
  bool sectionSelection{false};

  /// The most encoded size, as a fraction, that decode weighting may give up
  /// against what size-only selection would have chosen for the same
  /// column. Enforced wherever a decode-weighted choice is made: the split
  /// planner, the hybrid planner and a section's encoding selection. Inert
  /// at the default decode weight of zero.
  double maxSizeRegression{0.05};

  /// Chooses split boundaries with the hybrid planner instead of trusting
  /// the split DP's argmin. The DP serves as a shortlister; a small set of
  /// candidate plans is re-priced with section selection's own estimators
  /// and the winner is refined locally. Off by default; costs roughly twice
  /// the planning time.
  bool hybridPlanner{false};
};

} // namespace facebook::nimble::subintsplit
