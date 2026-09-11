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

#include <span>
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/BufferedEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Represents bools sparsely. Useful when only a small fraction  are set (or not
// set).
//
// In terms of space, this will be better than the trivial bitmap encoding when
// the fraction of set (or unset) bits is smaller than 1 / lg(n), where n is the
// number of rows in the stream. On normal size files thats around 5%.
//
// This encoding will likely be of most use in representing the nulls
// subencoding inside a NullableEncoding.
//
// TODO: Should we generalize the SparseEncoding idea to other data types?
// Maybe. Or maybe in our mainly-constant encoding implementation we simple use
// a SparseBoolEncoding on top of the dense encoded data.

namespace facebook::nimble {

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 1 byte: whether the sparse bits are set or unset
// XX bytes: indices encoding bytes
class SparseBoolEncoding final : public TypedEncoding<bool, bool> {
 public:
  using cppDataType = bool;

  SparseBoolEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  void materializeBoolsAsBits(uint32_t rowCount, uint64_t* buffer, int begin)
      final;

  /// Returns the bool value represented by the sparse position stream.
  bool sparseValue() const noexcept {
    return sparseValue_;
  }

  /// Appends sparse positions in the next rowCount rows, relative to the
  /// current row, and advances the row cursor.
  uint32_t materializeSparseIndices(
      uint32_t rowCount,
      Vector<uint32_t>& buffer);

  /// Advances rowCount rows and returns how many sparse positions were skipped.
  uint32_t skipSparseIndices(uint32_t rowCount);

  /// True counts before and inside a row range.
  struct RangeCounts {
    /// Number of true values before the range offset.
    uint32_t numTrueBeforeRange{0};

    /// Number of true values inside the range.
    uint32_t numTrueInRange{0};
  };

  /// Sliced SparseBool stream plus true counts needed by nested slicers.
  struct SliceResult {
    /// Encoded slice covering the requested row range.
    std::string_view sliced;

    /// True counts before and inside the requested row range.
    RangeCounts counts;
  };

  /// Fills true counts before and inside non-empty rows [offset, offset +
  /// length).
  static void countTrue(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      velox::memory::MemoryPool* pool,
      RangeCounts& counts,
      const Encoding::Options& options = {});

  /// Counts true values in non-empty rows [offset, offset + length).
  static uint32_t countTrue(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options = {});

  /// Slices rows [offset, offset + length) and counts true values before and
  /// inside the slice while walking sparse positions once.
  static SliceResult sliceAndCount(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options);

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<bool>& selection,
      std::span<const bool> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static uint64_t estimateSize(
      uint64_t rowCount,
      const Statistics<bool>& statistics,
      const Encoding::Options& options = {}) {
    // Assumptions:
    // Uncommon indices are stored bit-packed (with bit width capable of
    // representing max entry count).

    const auto exceptionCount = std::min(
        statistics.uniqueCounts().value().at(true),
        statistics.uniqueCounts().value().at(false));
    return estimateSize(rowCount, exceptionCount, options);
  }

  static uint64_t estimateSize(
      uint64_t rowCount,
      uint64_t exceptionCount,
      const Encoding::Options& options = {}) {
    // Sparse indices are encoded as a FixedBitWidth child.
    // A rowCount sentinel is appended after the real sparse positions so the
    // decoder can stop without storing an explicit index count.
    const uint64_t indicesEncodingSize =
        FixedBitWidthEncoding<uint32_t>::estimateSize(
            exceptionCount + 1,
            /*minValue=*/0,
            /*maxValue=*/rowCount,
            options);

    return EncodingPrefix::kFixedPrefixSize + kPrefixSize + indicesEncodingSize;
  }

 private:
  // Encodes a sliced SparseBool stream from sparse positions relative to the
  // slice start and with the row-count sentinel appended.
  static std::string_view encodeWithSlicedIndices(
      std::string_view encoded,
      uint32_t length,
      std::span<const uint32_t> slicedIndicesWithSentinel,
      Buffer& buffer,
      const Encoding::Options& options = {});

  // SparseBool stores one byte after the common encoding prefix to indicate
  // whether the sparse indices refer to set bits or unset bits.
  static constexpr int kPrefixSize = sizeof(uint8_t);

  // If true, then indices give the position of the set bits; if false they give
  // the positions of the unset bits.
  const bool sparseValue_;
  Vector<char> indicesUncompressed_;
  detail::BufferedEncoding<uint32_t, 32> indices_;
  uint32_t nextIndex_; // The current index (FBA value).
  uint32_t row_;
};

template <typename V>
void SparseBoolEncoding::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        while (nextIndex_ < row_) {
          nextIndex_ = indices_.nextValue();
        }
        if (FOLLY_UNLIKELY(nextIndex_ == row_++)) {
          nextIndex_ = indices_.nextValue();
          return sparseValue_;
        }
        return !sparseValue_;
      });
}

} // namespace facebook::nimble
