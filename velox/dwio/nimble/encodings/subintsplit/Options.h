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

#include "velox/dwio/nimble/encodings/subintsplit/DecodeCost.h"

namespace facebook::nimble::subintsplit {

/// How top-level selection decides whether SubIntSplit is tried. Under the
/// bit-flip modes an admitted stream is offered SubIntSplit as a candidate and
/// still has to win the ordinary size comparison; only
/// Options::admissionForces lets the gate decide alone.
enum class SubIntSplitAdmission : uint8_t {
  /// SubIntSplitEncoding::estimateSize competes with the other candidates on
  /// its read-factor-weighted size.
  kEstimate = 0,
  /// bitFlipGradientGate() alone decides whether SubIntSplit is a candidate.
  kBitFlip = 1,
  /// bitFlipGradientGate() and the active-bit entropy guard must both admit
  /// for SubIntSplit to be a candidate.
  kBitFlipEntropy = 2,
};

/// SubIntSplit's settings on Encoding::Options, held as
/// Encoding::Options::subIntSplit: those top-level selection consults about
/// SubIntSplit, and what SubIntSplit tells the selection of its own sections
/// (filled in by sectionEncodingOptions from its TuningConfig). SubIntSplit's
/// own planner and decoder settings live in TuningConfig.
struct Options {
  /// Whether these options are the ones a SubIntSplit section is being
  /// encoded with, rather than a column's own options. Marks the whole
  /// subtree below a section, so a nested stream is priced on decode too.
  bool sectionSelection{false};

  /// How much a section's decode cost counts against its encoded size, in
  /// bytes of encoded size per nanosecond per row of decode. Zero is
  /// size-only selection. See DecodeCost.h for the per-encoding rates.
  double decodeWeight{0.0};

  /// The read shape section decode is costed for when decodeWeight is
  /// non-zero.
  DecodeAccessPattern decodeAccessPattern{DecodeAccessPattern::Bulk};

  /// The reader section decode is costed for when decodeWeight is non-zero.
  DecodeReadPath decodeReadPath{DecodeReadPath::Cursor};

  /// The most encoded size, as a fraction, that decode weighting may give up
  /// against what size-only selection would have chosen.
  double maxSizeRegression{0.05};

  /// How top-level selection admits SubIntSplit. kEstimate, the default,
  /// keeps SubIntSplitEncoding::estimateSize competing with the other
  /// candidates. kBitFlip offers SubIntSplit as a candidate only when the
  /// bit-flip profile's gradient gate admits the stream; kBitFlipEntropy also
  /// requires the active-bit entropy guard (see TopLevelPolicy.h). An
  /// admitted stream still has to win the ordinary size comparison unless
  /// admissionForces is set.
  SubIntSplitAdmission admission{SubIntSplitAdmission::kEstimate};

  /// Whether a bit-flip admission decides on its own, rather than only which
  /// candidates compete. True makes an admitted stream SubIntSplit without a
  /// size comparison; kept for ablations that separate the gate's
  /// predictions from what selection does with them. Read only when
  /// admission is not kEstimate.
  bool admissionForces{false};

  /// Consecutive pairs the admission profile is computed over, taken at a
  /// fixed stride across the stream. 0 uses every pair. Read only when
  /// admission is not kEstimate. The default samples so the gate stays cheap
  /// enough to run before anything expensive.
  uint32_t admissionProfilePairs{1'024};

  /// Whether selection may choose SubIntSplit for a nested stream: an RLE's
  /// run values, a Dictionary's alphabet, a FrequencyPartition tier. True
  /// keeps the writer's candidate list; false restricts SubIntSplit to
  /// top-level selection. A SubIntSplit section never chooses SubIntSplit
  /// whatever this says.
  bool inNestedStreams{true};

  /// Whether the streams this selection writes are handed to a substream
  /// compressor once they are encoded. Set by the selection policy from the
  /// CompressionOptions it was built with, so SubIntSplit's size estimate can
  /// tell the two apart; a caller does not set it.
  bool substreamCompression{false};

  /// Whether SubIntSplit's estimate declines to price a split when
  /// substreamCompression says the stream will be compressed afterwards. The
  /// estimate ranks candidates on uncompressed bytes, which disagrees with
  /// ranking under a general-purpose compressor most where a split's
  /// uncompressed win is largest. A stop-gap for that mismatch, not a claim
  /// that a split never pays under a compressor. Uncompressed selection
  /// never reads this field.
  bool estimateCompressionGuard{true};

  /// Whether selection may rule SubIntSplit out from the bit-flip gradient
  /// gate rather than by planning a split (see
  /// SubIntSplitEncoding::estimateSizeLowerBound). Off by default: the gate's
  /// bound is a prediction rather than a proof, so it can withhold a
  /// candidate that would have won, and measured savings from skipping the DP
  /// are small. On for a writer that expects mostly negatives and wants to
  /// skip the DP on them.
  bool estimateBitFlipScreen{false};
};

} // namespace facebook::nimble::subintsplit
