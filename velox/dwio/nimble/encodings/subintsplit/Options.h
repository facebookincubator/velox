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

#include "velox/dwio/nimble/encodings/subintsplit/DecodeCost.h"

namespace facebook::nimble::subintsplit {

/// What SubIntSplit tells the encoding selection of its own sections, held by
/// Encoding::Options as Encoding::Options::subIntSplit. Filled in by
/// sectionEncodingOptions from SubIntSplit's TuningConfig, so selection weighs
/// a section's candidates the way the planner weighed its boundaries.
/// Callers leave it at its defaults; SubIntSplit's settings live in
/// TuningConfig.
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
};

} // namespace facebook::nimble::subintsplit
