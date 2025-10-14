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

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace facebook::velox::exec {

/// A minimal envelope for a geometry.
/// It also includes an index for the geometry for later reference.  This can
/// be -1 if the geometry is not indexed.
///
/// Envelopes use float32s instead of float64s so that SIMD loops can be
/// twice as fast.  Our geometries use float64 coordinates, so we have to
/// downcast them for the envelope.  The loss of precision is theoretically fine
/// because the envelope checks are already approximate: either they don't
/// intersect or they might intersect.  Thus, expanding the envelopes slightly
/// does not affect correctness (but it might affect efficiency slightly).
///
/// We want to show that if two envelopes expressed with float64 precision would
/// intersect, the envelopes with float32 precision would also intersect.
///
/// Define
/// ```
/// nextUp(f) = std::nextafter(f, std::numeric_limits<float>::infinity())
/// nextDown(f) = std::nextafter(f, -std::numeric_limits<float>::infinity())
/// ```
/// which move a float up or down one ulp (unit in the last place).
///
/// Since the conditions are all of the form `maxX >= minX` for float64s maxX
/// and minX, we need to show that this implies `nextUp((float) maxX) >=
/// nextDown((float) minX)`.
///
/// Assume you have a double `d` and two adjacent floats `f0` and `f1`, such
/// that `d` is "between" `f0` and `f1`:
///
/// 1. `(double) f0 <= d <= (double) f1`
/// 2. `nextup(f0) == f1 && f0 = nextdown(f0)`
///
/// This implies `nextdown((float) d) <= f0 && nextup((float) d) >= f1`.
///
/// Let double `minX` have two adjacent floats `f0`, `f1` as above, and `maxX`
/// have two adjacent floats `g0`, `g1`.  Then
/// ```
/// (double) nextDown((float) minX)
///   <= (double) f0
///   <= minX
///   <= maxX
///   <= (double) g1
///   <= (double) nextUp((float) maxX)
/// ```
///
/// And this implies `nextDown((float) minX) <= nextUp((float) maxX)` as
/// desired. The same argument applies to all for members, so if we construct
/// the float32 precision envelope by applying nextDown to the minX/Ys and
/// nextUp to maxX/Ys, the float32 envelope intersects in all cases that the
/// float64 envelope would (but not necessarily the converse).
struct Envelope {
  float minX;
  float minY;
  float maxX;
  float maxY;
  int32_t rowIndex = -1;

  static inline bool intersects(const Envelope& left, const Envelope& right) {
    return (left.maxX >= right.minX) && (left.minX <= right.maxX) &&
        (left.maxY >= right.minY) && (left.minY <= right.maxY);
  }

  inline bool isEmpty() const {
    return (maxX < minX) || (maxY < minY);
  }

  static constexpr inline Envelope empty() {
    return Envelope{
        .minX = std::numeric_limits<float>::infinity(),
        .minY = std::numeric_limits<float>::infinity(),
        .maxX = -std::numeric_limits<float>::infinity(),
        .maxY = -std::numeric_limits<float>::infinity()};
  }

  static constexpr inline Envelope from(
      double minX,
      double minY,
      double maxX,
      double maxY,
      int32_t rowIndex = -1) {
    return Envelope{
        .minX = std::nextafterf(
            static_cast<float>(minX), -std::numeric_limits<float>::infinity()),
        .minY = std::nextafterf(
            static_cast<float>(minY), -std::numeric_limits<float>::infinity()),
        .maxX = std::nextafterf(
            static_cast<float>(maxX), std::numeric_limits<float>::infinity()),
        .maxY = std::nextafterf(
            static_cast<float>(maxY), std::numeric_limits<float>::infinity()),
        .rowIndex = rowIndex};
  }
};

/// A spatial index for a set of geometries. The index only cares about the
/// envelopes of the geometries, and an index into the geometries (not stored in
/// SpatialIndex).
///
/// The contract is that SpatialIndex::probe returns the indices of all
/// envelopes that probeEnv intersects. The form of the index is an
/// implementation detail. The order of the returned indicies is an
/// implementation detail.
class SpatialIndex {
 public:
  SpatialIndex(const SpatialIndex&) = delete;
  SpatialIndex& operator=(const SpatialIndex&) = delete;

  SpatialIndex() = default;
  SpatialIndex(SpatialIndex&&) = default;
  SpatialIndex& operator=(SpatialIndex&&) = default;
  ~SpatialIndex() = default;

  explicit SpatialIndex(std::vector<Envelope> envelopes);

  /// Returns the row indices of all envelopes that probeEnv intersects.
  /// Order of the returned indices is an implementation detail and cannot be
  /// relied upon.
  std::vector<int32_t> query(const Envelope& queryEnv) const;

  /// Returns the envelope of the all envelopes in the index.
  /// The returned envelope will have index = -1.
  Envelope bounds() const;

 private:
  Envelope bounds_ = Envelope::empty();

  std::vector<double> minXs_{};
  std::vector<double> minYs_{};
  std::vector<double> maxXs_{};
  std::vector<double> maxYs_{};
  std::vector<int32_t> rowIndices_{};
};

} // namespace facebook::velox::exec
