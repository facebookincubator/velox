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

namespace facebook::nimble::subintsplit {

/// Settings encoding selection consults on SubIntSplit's behalf, held by
/// Encoding::Options as Encoding::Options::subIntSplit because selection sees
/// only those options. Callers leave them at their defaults; SubIntSplit sets
/// them for its own sections.
struct Options {
  /// When true, size estimates are tight enough to compare with another
  /// encoding's bytes rather than only to rank candidates. FixedBitWidth
  /// counts the slop bytes FixedBitArray reserves and its row count at the
  /// width the prefix stores it, MainlyConstant prices its other values
  /// over their own range and against Trivial, RLE prices
  /// its run values against Trivial and Dictionary and its run lengths with
  /// the encodings nested selection would pick, and selection withholds
  /// Trivial's read-factor discount where Trivial is larger than
  /// FixedBitWidth. SubIntSplit sets this for its sections, whose sizes it
  /// compares against each other; nothing else should.
  bool sectionEstimatorRefinements{false};
};

} // namespace facebook::nimble::subintsplit
