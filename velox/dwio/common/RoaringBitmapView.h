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

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace facebook::velox::dwio::common {

/// Read-only view of a Roaring bitmap of row positions in the portable
/// serialization, the form Iceberg and Paimon deletion vectors store. Parsing
/// checks the container headers and keeps each container serialized. apply()
/// sets the bits of the positions in a row range straight from the containers
/// it overlaps, without expanding every position.
///
/// The view does not own the serialized bytes, which must outlive it and must
/// not change.
class RoaringBitmapView {
 public:
  /// Serialization of the bitmap.
  enum class Format {
    /// A 32-bit RoaringBitmap in the RoaringFormatSpec portable layout.
    k32Bit,
    /// A Roaring64 bitmap: a little-endian 64-bit group count, then for each
    /// group in increasing order the high 32 bits of its positions and a
    /// 32-bit RoaringBitmap of their low 32 bits.
    k64Bit,
  };

  /// Largest Roaring64 group key, which holds the high 32 bits of a position.
  /// Iceberg's RoaringPositionBitmap caps the key one below INT32_MAX, and a
  /// key of 2^31 or more would shift into the sign bit of a position.
  static constexpr uint32_t kMaxGroupKey = 2'147'483'646;

  /// Creates an empty bitmap.
  RoaringBitmapView() = default;

  /// Parses the bitmap in 'data'. Throws if 'data' is not a valid bitmap of
  /// 'format'.
  RoaringBitmapView(std::string_view data, Format format);

  /// Sets bit 'position - begin' of 'bitmap' for every position in
  /// [begin, end). Calls must cover increasing, non-overlapping ranges.
  void apply(int64_t begin, int64_t end, uint8_t* bitmap);

  /// Returns true once apply() has passed every position.
  bool exhausted() const {
    return containerIndex_ >= containers_.size();
  }

  /// Returns every position in increasing order. Expands the whole bitmap, so
  /// scans use apply() instead.
  std::vector<int64_t> positions() const;

  /// Returns the number of positions. Counts the positions of every container
  /// on each call.
  uint64_t cardinality() const;

 private:
  // Encoding of one 64K-row container in the portable Roaring format.
  enum class ContainerKind : uint8_t { kArray, kBitset, kRun };

  // One Roaring container, kept in its serialized form.
  struct Container {
    // Encoding of the payload.
    ContainerKind kind;
    // Number of values for kArray, number of runs for kRun, unused for
    // kBitset.
    uint32_t count;
    // First row position the container covers.
    int64_t base;
    // Largest position in the container, or base - 1 if the container selects
    // nothing.
    int64_t lastPosition;
    // Offset of the container payload within the bitmap bytes.
    size_t dataOffset;
  };

  // Parses the groups of a Roaring64 bitmap.
  void parse64Bit();

  // Parses one 32-bit RoaringBitmap starting at 'offset' within the bitmap
  // bytes and returns the offset just past it. Positions are offset by
  // 'highBitsOffset', the upper 32 bits in 64-bit mode and 0 in 32-bit mode.
  size_t parse32Bit(size_t offset, int64_t highBitsOffset);

  const uint8_t* data() const {
    return reinterpret_cast<const uint8_t*>(data_.data());
  }

  // The serialized bitmap.
  std::string_view data_;

  // Containers in increasing row order.
  std::vector<Container> containers_;

  // First container that may still hold positions at or after the next range.
  size_t containerIndex_{0};
};

} // namespace facebook::velox::dwio::common
