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

#include <cstring>
#include <optional>
#include <span>
#include <type_traits>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

// The FixedBitWidthEncoding stores integer data in a fixed number of
// bits equal to the number of bits required to represent the largest value in
// the encoding. For now we only support encoding non-negative values, but
// we may later add an optional zigzag encoding that will let us handle
// negatives.

namespace facebook::nimble {

template <typename T>
class EncodingSelection;

/// Typed lanes of FixedBitWidthEncoding (see its data layout). They use the
/// bit-packed layout's rules with different parameters: the lane width rounds
/// the bit width up as the bit-packed width does (to a multiple of 8), but to
/// kNarrowLaneAlignmentBits or the type width, and the payload size is
/// FixedBitArray::bufferSize() rounded up to whole lanes.
struct FixedBitWidthTypedLanes {
  /// Minimum stored bit width of a chunk for typed lanes.
  static constexpr int kMinStoredBits{16};

  /// Narrow lanes round the bit width up to a multiple of this: 16- or 32-bit
  /// lanes.
  static constexpr int kNarrowLaneAlignmentBits{16};

  /// Returns the uncompressed payload size: whole lanes, and at least
  /// FixedBitArray::bufferSize().
  static uint64_t payloadSize(uint64_t rowCount, int laneBits) {
    return velox::bits::roundUp(
        FixedBitArray::bufferSize(rowCount, laneBits), laneBits / 8);
  }

  /// Returns whether 'compressionType' reads the payload's DataType. Only such
  /// compressors gain from lanes: Managed Compression (MetaInternal) and
  /// OpenZL encode them as integers. A byte-oriented compressor (Zstd, Lz4)
  /// would only see a wider payload, and an uncompressed payload would only be
  /// stored wider. A compression type added later takes no lanes until it is
  /// listed here.
  static constexpr bool compressorDataTypeAware(
      CompressionType compressionType) {
    return compressionType == CompressionType::MetaInternal ||
        compressionType == CompressionType::OpenZL;
  }
};

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 1 byte: compression type
// sizeof(T) byte: baseline value
// 1 byte: bit width
// FixedBitArray::BufferSize(rowCount, bit_width) bytes: packed values.
//
// Typed lanes (Encoding::Options::fixedBitWidthTypedLanes) are the same layout
// with a payload of FixedBitArray::bufferSize() rounded up to whole lanes,
// exactly rowCount * bit_width / 8 + 8 bytes (FixedBitWidthTypedLanes),
// compressed with a typed DataType instead of Undefined:
// - Narrow lanes, for 32- and 64-bit types whose stored width is at most half
//   the type width: a bit width of 16 (values of up to 16 bits) or 32, the
//   minimum as baseline, compressed as Uint16 or Uint32.
// - Full-width lanes, for every other eligible chunk (including every eligible
//   16-bit chunk): the type's bit width and the minimum as baseline,
//   compressed as the stream's DataType, the label TrivialEncoding uses.
// The padding keeps typed compressors on whole lanes: Managed Compression's
// typed graphs take only whole lanes, and in OpenZL's permissive mode, its
// default, a rejected size is stored as zstd without an error. It also covers
// FixedBitArray's reads of whole 64-bit words, up to 7 bytes past the first
// byte of the last value.
// Readers need no option for typed lanes: FixedBitArray does not depend on the
// payload length, and compressed frames are self-describing, so decompressing
// them as Undefined returns the same bytes.
template <typename T>
class FixedBitWidthEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  FixedBitWidthEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  ~FixedBitWidthEncoding() override {
    this->releaseBuffer(uncompressedData_);
  }

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  // Bulk scan method for fast path decoding.
  // Reads multiple values at once and processes them through the visitor.
  // This is used by readWithVisitorFast for efficient batch processing.
  template <bool kScatter, typename Visitor>
  void bulkScan(
      Visitor& visitor,
      vector_size_t currentRow,
      const vector_size_t* selectedRows,
      vector_size_t numSelected,
      const vector_size_t* scatterRows);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  // TODO: If compressed sorted-position sub-streams become common in
  // production or profiling shows the per-slice EncodingView allocation and
  // virtual dispatch measurable on partial-stripe reads, add an FBW-specific
  // fast path that shares a decompressed payload between the value-range
  // binary search and the slice extraction. Skipped for now because
  // production writers do not compress sorted-position sub-streams today.
  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static uint64_t estimateSize(
      uint64_t rowCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    return estimateSize(rowCount, statistics.min(), statistics.max(), options);
  }

  static uint64_t estimateSize(
      uint64_t rowCount,
      uint64_t minValue,
      uint64_t maxValue,
      const Encoding::Options& options) {
    const uint64_t outerEncodingSize =
        EncodingPrefix::kFixedPrefixSize + kPrefixSize;
    const uint64_t payloadSize = bitPackedBytes(
        minValue, maxValue, rowCount, options.fixedBitWidthUseExactBits);
    return outerEncodingSize + payloadSize;
  }

  std::string debugString(int offset) const final;

 private:
  static constexpr int kPrefixSize = 2 + sizeof(T);

  static uint64_t bitPackedBytes(
      uint64_t minValue,
      uint64_t maxValue,
      uint64_t count,
      bool useExactBitWidth) {
    auto bitWidth = velox::bits::bitsRequired(maxValue - minValue);
    if (!useExactBitWidth) {
      bitWidth = velox::bits::roundUp(bitWidth, 8);
    }
    return velox::bits::nbytes(bitWidth * count);
  }

  // The typed-lane decision, made only here: whether a chunk of 'numValues'
  // values whose bit-packed width is 'storedBits' is stored as typed lanes
  // under 'options' when compressed with 'compressionType'.
  // - Only chunks encoded with an integer type take lanes. FixedBitWidth of
  //   float or double keeps the bit-packed layout; values stored as integers,
  //   such as the non-null values of a nullable float or double stream, can
  //   take lanes.
  // - Lanes widen the uncompressed payload. They pay off only for a compressor
  //   that reads the payload's DataType and so sees whole integers
  //   (FixedBitWidthTypedLanes::compressorDataTypeAware()).
  // 'compressionType' is the chunk's compressor. A caller without one passes
  // nullopt, and the compressor condition is then not checked.
  static bool takesTypedLanes(
      uint64_t numValues,
      int storedBits,
      std::optional<CompressionType> compressionType,
      const Encoding::Options& options) {
    if constexpr (!isIntegralType<T>()) {
      return false;
    } else {
      return options.fixedBitWidthTypedLanes &&
          storedBits >= FixedBitWidthTypedLanes::kMinStoredBits &&
          numValues >= options.fixedBitWidthTypedLanesMinValues &&
          (!compressionType.has_value() ||
           FixedBitWidthTypedLanes::compressorDataTypeAware(*compressionType));
    }
  }

  // Returns the lane width for a chunk of 'rowCount' values needing
  // 'exactBits' bits and stored in 'storedBits', compressed with 'policy', or
  // 0 when the chunk is bit-packed (see the data layout).
  static int typedLaneBits(
      uint32_t rowCount,
      int exactBits,
      int storedBits,
      const CompressionPolicy& policy,
      const Encoding::Options& options);

  int bitWidth_;
  physicalType baseline_;
  FixedBitArray fixedBitArray_;
  uint32_t row_ = 0;
  velox::BufferPtr uncompressedData_;
};

//
// End of class declaration. Implementations follow.
//

template <typename T>
FixedBitWidthEncoding<T>::FixedBitWidthEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> /* stringBufferFactory */,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options} {
  auto pos = data.data() + this->dataOffset();
  auto compressionType = static_cast<CompressionType>(encoding::readChar(pos));
  baseline_ = encoding::read<const physicalType>(pos);
  bitWidth_ = static_cast<uint32_t>(encoding::readChar(pos));
  if (compressionType != CompressionType::Uncompressed) {
    uncompressedData_ = Compression::uncompress(
        pool,
        compressionType,
        DataType::Undefined,
        {pos, static_cast<size_t>(data.data() + data.size() - pos)},
        options.decompressCounter(),
        options.bufferPool);
    fixedBitArray_ = FixedBitArray{
        {uncompressedData_->as<char>(), uncompressedData_->size()}, bitWidth_};
  } else {
    fixedBitArray_ = FixedBitArray{
        {pos, static_cast<size_t>(data.data() + data.size() - pos)}, bitWidth_};
  }
}

template <typename T>
void FixedBitWidthEncoding<T>::reset() {
  row_ = 0;
}

template <typename T>
void FixedBitWidthEncoding<T>::skip(uint32_t rowCount) {
  row_ += rowCount;
}

template <typename T>
void FixedBitWidthEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  fixedBitArray_.bulkGetWithBaseline(
      row_, rowCount, static_cast<physicalType*>(buffer), baseline_);
  row_ += rowCount;
}

template <typename T>
template <typename V>
void FixedBitWidthEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  using OutputType = detail::ValueType<typename V::DataType>;
  constexpr bool kIsSuitableWidth =
      (isFourByteIntegralType<physicalType>() ||
       isEightByteIntegralType<physicalType>());
  constexpr bool kIsFluidCast = sizeof(OutputType) >= sizeof(physicalType) &&
      std::is_integral_v<OutputType> && std::is_integral_v<physicalType>;
  // Fast path: use bulk scan for 4-byte and 8-byte integral types.
  // Compile-time: type constraints (FBW physical type, compatible integral
  // output type at least as wide as the physical type). Runtime: scanFastPath
  // unless bulkFastPathByDefault, then useFastPath (deterministic filter,
  // AVX2, bulk path enabled, null+filter/hook compatibility).
  if constexpr (kIsSuitableWidth && kIsFluidCast) {
    auto* nulls = visitor.reader().rawNullsInReadRange();
    if ((detail::bulkFastPathByDefault<V>() || params.scanFastPath) &&
        velox::dwio::common::useFastPath(visitor, nulls)) {
      detail::readWithVisitorFast(*this, visitor, params, nulls);
      return;
    }
  }
  // Slow path: process one value at a time.
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        physicalType value = fixedBitArray_.get(row_++) + baseline_;
        return value;
      });
}

template <typename T>
template <bool kScatter, typename V>
void FixedBitWidthEncoding<T>::bulkScan(
    V& visitor,
    vector_size_t currentRow,
    const vector_size_t* selectedRows,
    vector_size_t numSelected,
    const vector_size_t* scatterRows) {
  using OutputType = detail::ValueType<typename V::DataType>;
  static_assert(
      isFourByteIntegralType<physicalType>() ||
          isEightByteIntegralType<physicalType>(),
      "bulkScan only supports 4-byte or 8-byte integral types");

  if (numSelected == 0) {
    return;
  }

  const auto numRows = visitor.numRows() - visitor.rowIndex();

  // Calculate offset between our internal position and the external row number.
  // This handles cases where the encoding position (row_) differs from the
  // logical row number (currentRow).
  const auto offset =
      static_cast<int32_t>(row_) - static_cast<int32_t>(currentRow);

  // Get the output buffer.
  auto* values = detail::mutableValues<OutputType>(visitor, numRows);

  // Check type compatibility:
  // - Same type or same-size integral types: use fast memcpy
  //   (e.g., uint32_t vs int32_t have same bit representation)
  // - Widening (e.g., int32 → int64): use loop with implicit conversion
  constexpr bool kSameSize = sizeof(physicalType) == sizeof(OutputType);
  constexpr bool kIsUpcast = sizeof(OutputType) > sizeof(physicalType) &&
      std::is_integral_v<OutputType> && std::is_integral_v<physicalType>;

  if constexpr (V::dense) {
    if constexpr (isFourByteIntegralType<physicalType>()) {
      // 4-byte path: use the optimized template-unrolled bulk decode.
      if constexpr (kSameSize) {
        fixedBitArray_.bulkGetWithBaseline(
            selectedRows[0] + offset,
            numSelected,
            reinterpret_cast<physicalType*>(values),
            baseline_);
      } else if constexpr (kIsUpcast) {
        static_assert(isEightByteIntegralType<OutputType>());
        // Read the 4-byte-width values directly into the wider 8-byte output in
        // a single pass, avoiding the staging buffer and per-element widening.
        fixedBitArray_.bulkGetWithBaseline32Into64(
            selectedRows[0] + offset,
            numSelected,
            reinterpret_cast<uint64_t*>(values),
            baseline_);
      }
    } else {
      // 8-byte path: use bulkGet64WithBaseline which handles all bit widths
      // including branchless byte-aligned loads for bitWidth <= 58.
      static_assert(isEightByteIntegralType<physicalType>());
      static_assert(kSameSize, "8-byte bulkScan requires same-size output");
      fixedBitArray_.bulkGetWithBaseline(
          selectedRows[0] + offset,
          numSelected,
          reinterpret_cast<physicalType*>(values),
          baseline_);
    }
  } else {
    // Sparse case: read individual values at specified positions.
    for (vector_size_t i = 0; i < numSelected; ++i) {
      values[i] = static_cast<OutputType>(
          fixedBitArray_.get(selectedRows[i] + offset) + baseline_);
    }
  }

  row_ += selectedRows[numSelected - 1] - currentRow + 1;

  // No scatter, filter, or hook: values are already in the output buffer.
  if constexpr (!kScatter && !V::kHasFilter && !V::kHasHook) {
    visitor.addNumValues(numRows);
    visitor.setRowIndex(visitor.numRows());
    return;
  }

  detail::applyFixedWidthRun<kScatter>(
      visitor, selectedRows, numSelected, scatterRows, values, numRows);
  visitor.setRowIndex(visitor.numRows());
}

template <typename T>
std::string_view FixedBitWidthEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  static_assert(
      std::is_same_v<
          typename std::make_unsigned<physicalType>::type,
          physicalType>,
      "Physical type must be unsigned.");
  const uint32_t rowCount = values.size();
  const int exactBitsRequired = velox::bits::bitsRequired(
      selection.statistics().max() - selection.statistics().min());
  const int storedBitsRequired = options.fixedBitWidthUseExactBits
      ? exactBitsRequired
      : velox::bits::roundUp(exactBitsRequired, 8);

  auto dataCompressionPolicy = selection.compressionPolicy();
  const int laneBits = typedLaneBits(
      rowCount,
      exactBitsRequired,
      storedBitsRequired,
      *dataCompressionPolicy,
      options);
  const bool fullWidthLanes =
      laneBits == 8 * static_cast<int>(sizeof(physicalType));

  // typedLaneBits returns 0 (bit-packed), a narrow lane (16 or 32 bits), or the
  // type width (full-width lanes); the payload DataType below relies on that.
  NIMBLE_DCHECK(
      laneBits == 0 || fullWidthLanes || laneBits == 16 || laneBits == 32,
      "Unexpected typed lane width: {}",
      laneBits);
  const int bitsRequired = laneBits > 0 ? laneBits : storedBitsRequired;
  const physicalType baseline = selection.statistics().min();
  const uint32_t fixedBitArraySize = laneBits > 0
      ? FixedBitWidthTypedLanes::payloadSize(rowCount, laneBits)
      : FixedBitArray::bufferSize(values.size(), bitsRequired);

  // Typed lanes compress the payload as Uint16, Uint32 or the stream's
  // DataType, and the header does not record which. Decompression does not
  // need it. OpenZL frames describe how to decode them. Managed Compression
  // picks the decoder from the frame's magic bytes, and its dictionaries and
  // OpenZL transforms are not tied to a category. So its fallback category,
  // which Undefined selects, decodes frames written by the u16, u32 and u64
  // categories. That requires the fallback category to keep OpenZL enabled.
  const DataType payloadDataType = laneBits == 0 ? DataType::Undefined
      : fullWidthLanes                           ? TypeTraits<T>::dataType
      : laneBits == 16                           ? DataType::Uint16
                                                 : DataType::Uint32;

  Vector<char> vector{&buffer.getMemoryPool()};

  CompressionEncoder<T> compressionEncoder{
      buffer.getMemoryPool(),
      *dataCompressionPolicy,
      payloadDataType,
      bitsRequired,
      fixedBitArraySize,
      [&]() {
        vector.resize(fixedBitArraySize);
        return std::span<char>{vector};
      },
      [&](char*& pos) {
        memset(pos, 0, fixedBitArraySize);
        FixedBitArray fba(pos, bitsRequired);
        fba.bulkSetWithBaseline(0, rowCount, values.data(), baseline);
        pos += fixedBitArraySize;
        return pos;
      }};

  const uint32_t encodingSize =
      Encoding::serializePrefixSize(rowCount, useVarint) +
      FixedBitWidthEncoding<T>::kPrefixSize + compressionEncoder.getSize();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::FixedBitWidth,
      TypeTraits<T>::dataType,
      rowCount,
      useVarint,
      pos);
  encoding::writeChar(
      static_cast<char>(compressionEncoder.compressionType()), pos);
  encoding::write(baseline, pos);
  encoding::writeChar(bitsRequired, pos);
  compressionEncoder.write(pos);

  NIMBLE_DCHECK_EQ(encodingSize, pos - reserved, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
int FixedBitWidthEncoding<T>::typedLaneBits(
    uint32_t rowCount,
    int exactBits,
    int storedBits,
    const CompressionPolicy& policy,
    const Encoding::Options& options) {
  if (!takesTypedLanes(
          rowCount, storedBits, policy.config().compressionType, options)) {
    return 0;
  }

  // The bit width rounds up as for bit-packed chunks, to a different
  // multiple: narrow lanes, for 32- and 64-bit types whose stored width is at
  // most half the type width, to 16 bits; every other chunk to the type width.
  constexpr int kTypeBits = 8 * sizeof(physicalType);
  const int alignmentBits = (2 * storedBits <= kTypeBits)
      ? FixedBitWidthTypedLanes::kNarrowLaneAlignmentBits
      : kTypeBits;
  return velox::bits::roundUp(exactBits, alignmentBits);
}

template <typename T>
std::string_view FixedBitWidthEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  EncodingPrefix::checkSliceRange(
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount),
      offset,
      length);

  const auto sourcePrefixSize =
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const char* sourcePos = encoded.data() + sourcePrefixSize;
  const auto sourceCompressionType =
      static_cast<CompressionType>(encoding::readChar(sourcePos));
  const auto baseline = encoding::read<physicalType>(sourcePos);
  const auto bitWidth = static_cast<uint8_t>(encoding::readChar(sourcePos));

  velox::BufferPtr uncompressed;
  std::string_view packedData{
      sourcePos,
      static_cast<size_t>(encoded.data() + encoded.size() - sourcePos)};
  if (sourceCompressionType != CompressionType::Uncompressed) {
    uncompressed = Compression::uncompress(
        buffer.getMemoryPool(),
        sourceCompressionType,
        DataType::Undefined,
        packedData,
        options.decompressCounter(),
        options.bufferPool);
    packedData = {uncompressed->as<char>(), uncompressed->size()};
  }

  const auto packedBytes = FixedBitArray::bufferSize(length, bitWidth);
  const auto prefixSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount);
  const auto encodingSize =
      prefixSize + FixedBitWidthEncoding<T>::kPrefixSize + packedBytes;
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  EncodingPrefix::serialize(
      EncodingType::FixedBitWidth,
      TypeTraits<T>::dataType,
      length,
      options.useVarintRowCount,
      pos);
  encoding::writeChar(static_cast<char>(CompressionType::Uncompressed), pos);
  encoding::write(baseline, pos);
  encoding::writeChar(bitWidth, pos);

  // FixedBitArray::bufferSize() always adds slop bytes, so the payload is never
  // empty even at zero bit width. Zeroing keeps the slop deterministic; the bit
  // copy is skipped at zero bit width because there is nothing to copy and
  // copyPackedBits() rejects a zero bit count.
  std::memset(pos, 0, packedBytes);
  if (bitWidth > 0) {
    const auto sourceBitOffset = static_cast<uint64_t>(offset) * bitWidth;
    const auto sliceBits = static_cast<uint64_t>(length) * bitWidth;
    encoding::copyPackedBits(packedData, sourceBitOffset, sliceBits, pos);
  }
  pos += packedBytes;

  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string FixedBitWidthEncoding<T>::debugString(int offset) const {
  return fmt::format(
      "{}{}<{}> rowCount={} bit_width={}",
      std::string(offset, ' '),
      toString(Encoding::encodingType()),
      toString(Encoding::dataType()),
      Encoding::rowCount(),
      bitWidth_);
}

} // namespace facebook::nimble
