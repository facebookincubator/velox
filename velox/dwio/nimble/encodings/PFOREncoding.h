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

#include <algorithm>
#include <cstring>
#include <span>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// PFOREncoding stores integer data using Patched Frame-of-Reference (slot 15).
// Each value is decomposed as `value = baseline + residual`, where the
// residuals are bitpacked at a base bit width chosen so that ~90% of the
// residuals fit. Values whose residual overflows the base width are recorded
// as (position, value) "exceptions" stored separately as two self-describing
// nested sub-encodings (one for the positions, one for the values), each
// chosen by Nimble's recursive encoding selection.
//
// Decode uses a branchless two-pass strategy (ported from AusIntListPfor):
//   Pass 1: Unpack all base residuals in a tight branchless loop.
//   Pass 2: Patch the ~10% exception positions with their full values.
// This eliminates per-element branches from the hot loop.
//
// Pfor wins over plain FixedBitWidth on dense numeric streams that contain a
// small fraction of large outliers -- FixedBitWidth would have to widen every
// slot to the worst case, while Pfor pays the wide cost only for the outliers.

namespace facebook::nimble {

/// Data layout (after the standard Encoding prefix):
///
///   sizeof(physicalType) bytes : baseline (a.k.a. min value)
///   1 byte                    : baseBitWidth (bits per bitpacked residual;
///                               range [0, 64])
///   varint                    : numExceptions
///   varint + N bytes          : exception positions sub-stream -- a varint
///                               size prefix followed by a nested encoding of
///                               the strictly ascending positions (size 0 and
///                               no encoding when there are no exceptions)
///   varint + N bytes          : exception values sub-stream -- a varint size
///                               prefix followed by a nested encoding of the
///                               full residuals, i.e. value - baseline (size 0
///                               and no encoding when there are no exceptions)
///   FixedBitArray::bufferSize(rowCount, baseBitWidth) bytes:
///                               bitpacked base residuals (omitted when
///                               baseBitWidth == 0)
template <typename T>
class PFOREncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  PFOREncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  ~PFOREncoding() override {
    this->releaseVectorBuffer(exceptionValues_);
    this->releaseVectorBuffer(exceptionPositions_);
  }

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  std::string debugString(int offset) const final;

  static uint64_t estimateSize(
      uint64_t rowCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options = {}) {
    const auto fullRange =
        static_cast<physicalType>(statistics.max() - statistics.min());
    const uint8_t maxBitWidth =
        static_cast<uint8_t>(velox::bits::bitsRequired(fullRange));
    const auto [baseBitWidth, numExceptions] = selectBaseBitWidth(
        statistics.bucketCounts(),
        static_cast<uint32_t>(rowCount),
        maxBitWidth);
    const uint64_t baseValuesSize = baseBitWidth == 0
        ? 0
        : FixedBitArray::bufferSize(rowCount, baseBitWidth);
    const uint64_t positionsSize = std::min(
        TrivialEncoding<uint32_t>::estimateSize(numExceptions),
        FixedBitWidthEncoding<uint32_t>::estimateSize(
            numExceptions,
            /*minValue=*/0,
            /*maxValue=*/rowCount == 0 ? 0 : rowCount - 1,
            options));
    const uint64_t valuesSize = std::min(
        TrivialEncoding<physicalType>::estimateSize(numExceptions),
        FixedBitWidthEncoding<physicalType>::estimateSize(
            numExceptions, statistics, options));
    const uint64_t outerEncodingSize = EncodingPrefix::kFixedPrefixSize +
        kFixedHeaderSize + varint::varintSize(numExceptions) +
        varint::varintSize(positionsSize) + varint::varintSize(valuesSize);
    return outerEncodingSize + baseValuesSize + positionsSize + valuesSize;
  }

 private:
  // Selects the base bit width from the bucket histogram, targeting 90%
  // coverage. Returns {baseBitWidth, numExceptions}.
  // Shared by encode() and EncodingSizeEstimation to keep the two in sync.
  //
  // Note: buckets have 7-bit granularity, so the selected baseBitWidth may
  // overestimate by up to 6 bits. The encode() path may tighten the serialized
  // base width after the actual residual pass when exact bit-width mode is
  // enabled.
  template <typename BucketArray>
  static std::pair<uint8_t, uint64_t> selectBaseBitWidth(
      const BucketArray& bucketCounts,
      uint32_t rowCount,
      uint8_t maxBitWidth) {
    constexpr double kCoverageThreshold = 0.9;
    if (maxBitWidth == 0) {
      return {0, 0};
    }
    const uint64_t threshold = static_cast<uint64_t>(
        static_cast<double>(rowCount) * kCoverageThreshold);
    uint64_t cumulative = 0;
    for (size_t k = 0; k < bucketCounts.size(); ++k) {
      cumulative += bucketCounts[k];
      if (cumulative >= threshold) {
        const uint8_t bucketEndBitWidth =
            static_cast<uint8_t>(std::min<size_t>((k + 1) * 7, 64));
        return {
            std::min(bucketEndBitWidth, maxBitWidth), rowCount - cumulative};
      }
    }
    return {maxBitWidth, 0};
  }

  /// Fixed-size part of the PFOR header: baseline + baseBitWidth(1 byte).
  static constexpr int kFixedHeaderSize =
      sizeof(physicalType) + sizeof(uint8_t);

  // Sliced exception child streams plus the exception count written to the
  // sliced PFOR header.
  struct ExceptionSlice {
    std::string_view positions;
    std::string_view values;
    uint32_t count{0};
  };

  static ExceptionSlice sliceExceptions(
      std::string_view exceptionPositions,
      std::string_view exceptionValues,
      uint32_t numExceptions,
      uint32_t offset,
      uint32_t length,
      velox::memory::MemoryPool* pool,
      Buffer& scratchBuffer,
      const Encoding::Options& options);

  // Patches any exceptions whose absolute position falls inside
  // [row_, row_ + count) onto `output`, where output[k] corresponds to
  // absolute position row_ + k. Advances exceptionCursor_ past consumed
  // exceptions.
  void patchExceptions(uint32_t count, physicalType* output);

  // Advances exceptionCursor_ past any exception positions that fall before
  // `targetRow`. Uses binary search for efficient seeking during skip()
  // and reset().
  void seekExceptionsTo(uint32_t targetRow);

  // Wire-format header values, populated in the constructor.
  physicalType baseline_{};
  uint8_t baseBitWidth_{0};
  uint32_t numExceptions_{0};

  // Decoded exception side-channel. Eagerly decoded and stored at
  // construction time since numExceptions_ is bounded by ~10% of rowCount
  // in the typical case. Positions are in strictly ascending order.
  Vector<uint32_t> exceptionPositions_;
  Vector<physicalType> exceptionValues_;

  // Bitpacked base-residual region. Default-constructed (empty) when
  // baseBitWidth_ == 0.
  FixedBitArray fixedBitArray_{};

  // Current absolute row position in the stream.
  uint32_t row_{0};

  // Index of the next unconsumed exception in exceptionPositions_ /
  // exceptionValues_. Monotonically increases as rows are consumed.
  uint32_t exceptionCursor_{0};
};

//
// End of class declaration. Implementations follow.
//

template <typename T>
PFOREncoding<T>::PFOREncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      exceptionPositions_{this->template getVectorBuffer<uint32_t>()},
      exceptionValues_{this->template getVectorBuffer<physicalType>()} {
  if constexpr (!isIntegralType<physicalType>()) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "Pfor encoding only supports integral data types.");
  } else {
    const char* pos = data.data() + this->dataOffset();

    // Parse the header: baseline, baseBitWidth, numExceptions.
    baseline_ = encoding::read<physicalType>(pos);
    baseBitWidth_ = static_cast<uint8_t>(encoding::readChar(pos));
    NIMBLE_CHECK_LE(
        baseBitWidth_, 64, "Pfor base bit width must be in [0, 64].");
    numExceptions_ = varint::readVarint32(&pos);
    NIMBLE_CHECK_LE(
        numExceptions_,
        this->rowCount_,
        "Pfor exception count exceeds row count.");

    auto readExceptionStream = [&](auto& subStream) {
      subStream.resize(numExceptions_);
      const uint32_t size = varint::readVarint32(&pos);
      if (numExceptions_ == 0) {
        NIMBLE_CHECK_EQ(size, 0, "Empty Pfor exception stream has data.");
      } else {
        auto subEncoding = EncodingFactory(options).create(
            pool, {pos, size}, stringBufferFactory);
        subEncoding->materialize(numExceptions_, subStream.data());
      }
      pos += size;
    };
    readExceptionStream(exceptionPositions_); // exception positions
    readExceptionStream(exceptionValues_); // exception values

    NIMBLE_CHECK_EQ(
        exceptionPositions_.size(),
        exceptionValues_.size(),
        "Pfor exception positions and values must have the same size.");

    // Validate the materialized positions (strictly ascending, in range).
    for (uint32_t i = 0; i < numExceptions_; ++i) {
      NIMBLE_CHECK_LT(
          exceptionPositions_[i],
          this->rowCount_,
          "Pfor exception position out of range.");
      if (i > 0) {
        NIMBLE_CHECK_GT(
            exceptionPositions_[i],
            exceptionPositions_[i - 1],
            "Pfor exception positions must be strictly ascending.");
      }
    }

    // Map the bitpacked residual region. Verify the wire data has
    // enough bytes remaining to prevent out-of-bounds reads during decode.
    // baseBitWidth_ == 0 when all values are identical (range == 0).
    // No bitpacked region is stored in that case.
    if (baseBitWidth_ > 0) {
      const size_t bitpackedSize =
          FixedBitArray::bufferSize(this->rowCount_, baseBitWidth_);
      NIMBLE_CHECK_GE(
          static_cast<size_t>(data.end() - pos),
          bitpackedSize,
          "Pfor bitpacked region underruns wire data.");
      fixedBitArray_ = FixedBitArray{
          {pos, static_cast<size_t>(data.end() - pos)}, baseBitWidth_};
    }
  }
  reset();
}

template <typename T>
void PFOREncoding<T>::reset() {
  row_ = 0;
  exceptionCursor_ = 0;
}

template <typename T>
void PFOREncoding<T>::seekExceptionsTo(uint32_t targetRow) {
  // Linear scan for small exception counts, binary search for large counts.
  const uint32_t remaining = numExceptions_ - exceptionCursor_;
  constexpr uint32_t kLinearScanThreshold{64};
  if (remaining < kLinearScanThreshold) {
    while (exceptionCursor_ < numExceptions_ &&
           exceptionPositions_[exceptionCursor_] < targetRow) {
      ++exceptionCursor_;
    }
  } else {
    const auto begin = exceptionPositions_.begin();
    const auto it = std::lower_bound(
        begin + exceptionCursor_, exceptionPositions_.end(), targetRow);
    exceptionCursor_ = static_cast<uint32_t>(it - begin);
  }
}

template <typename T>
void PFOREncoding<T>::skip(uint32_t rowCount) {
  row_ += rowCount;
  seekExceptionsTo(row_);
}

template <typename T>
void PFOREncoding<T>::patchExceptions(uint32_t count, physicalType* output) {
  // Overwrite exception slots with their full values. Exceptions are
  // in ascending position order, so this is a forward linear scan.
  const auto endRow = row_ + count;
  while (exceptionCursor_ < numExceptions_ &&
         exceptionPositions_[exceptionCursor_] < endRow) {
    const auto row = exceptionPositions_[exceptionCursor_];
    output[row - row_] = static_cast<physicalType>(
        baseline_ + exceptionValues_[exceptionCursor_]);
    ++exceptionCursor_;
  }
}

template <typename T>
void PFOREncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  if (rowCount == 0) {
    return;
  }

  // Unpack all base residuals (branchless bulk decode).
  auto* output = static_cast<physicalType*>(buffer);
  if (baseBitWidth_ == 0) {
    // All residuals are zero; every value equals baseline.
    std::fill(output, output + rowCount, baseline_);
  } else {
    fixedBitArray_.bulkGetWithBaseline(row_, rowCount, output, baseline_);
  }

  // Patch exception positions with their full values.
  patchExceptions(rowCount, output);
  row_ += rowCount;
}

template <typename T>
template <typename V>
void PFOREncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  // TODO: Add a bulkScan / readWithVisitorFast path for 4-byte and 8-byte
  // integral types once Pfor is used on hot read paths. The challenge is
  // interleaving exception patching with the bulk scan framework. For now
  // the slow path is sufficient since Pfor is a niche encoding.
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&]() {
        physicalType value;
        if (baseBitWidth_ == 0) {
          value = baseline_;
        } else {
          value =
              static_cast<physicalType>(fixedBitArray_.get(row_) + baseline_);
        }
        // Override with the exception value if one lands exactly on this row.
        if (exceptionCursor_ < numExceptions_ &&
            exceptionPositions_[exceptionCursor_] == row_) {
          value = static_cast<physicalType>(
              baseline_ + exceptionValues_[exceptionCursor_]);
          ++exceptionCursor_;
        }
        ++row_;
        return value;
      });
}

template <typename T>
std::string_view PFOREncoding<T>::encode(
    EncodingSelection<typename TypeTraits<T>::physicalType>& selection,
    std::span<const typename TypeTraits<T>::physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  if constexpr (!isIntegralType<physicalType>()) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "Pfor encoding only supports integral data types.");
  }

  static_assert(
      std::is_same_v<
          typename std::make_unsigned<physicalType>::type,
          physicalType>,
      "Pfor physical type must be unsigned.");

  const bool useVarint = options.useVarintRowCount;
  NIMBLE_CHECK(!values.empty(), "Pfor encoding cannot be used with 0 rows.");

  const uint32_t rowCount = static_cast<uint32_t>(values.size());
  const physicalType baseline = selection.statistics().min();
  const physicalType maxValue = selection.statistics().max();
  const physicalType fullRange = static_cast<physicalType>(maxValue - baseline);
  const uint8_t maxBitWidth =
      static_cast<uint8_t>(velox::bits::bitsRequired(fullRange));

  const auto [selectedBaseBitWidth, expectedExceptions] = selectBaseBitWidth(
      selection.statistics().bucketCounts(), rowCount, maxBitWidth);

  // Single pass: compute residuals, identify exceptions that overflow
  // baseBitWidth, and zero-mask exception slots in the residual array.
  constexpr uint32_t kBitsPerPhysicalType = sizeof(physicalType) * 8;
  const physicalType baseMask = selectedBaseBitWidth == 0
      ? physicalType{0}
      : static_cast<physicalType>(velox::bits::lowMask(selectedBaseBitWidth));

  auto* pool = &buffer.getMemoryPool();
  Vector<uint32_t> exceptionPositions{pool};
  Vector<physicalType> exceptionValues{pool};
  // Reserve for expected exceptions based on the 90% coverage threshold.
  exceptionPositions.reserve(expectedExceptions);
  exceptionValues.reserve(expectedExceptions);

  Vector<physicalType> maskedResiduals{pool};
  maskedResiduals.resize(rowCount);
  physicalType maxBaseResidual{0};
  for (auto i = 0; i < rowCount; ++i) {
    const physicalType residual =
        static_cast<physicalType>(values[i] - baseline);
    if (selectedBaseBitWidth < kBitsPerPhysicalType && residual > baseMask) {
      exceptionPositions.emplace_back(i);
      exceptionValues.emplace_back(residual);
      maskedResiduals[i] = physicalType{0};
    } else {
      maskedResiduals[i] = residual;
      maxBaseResidual = std::max(maxBaseResidual, residual);
    }
  }
  const uint32_t numExceptions =
      static_cast<uint32_t>(exceptionPositions.size());
  const uint8_t exactBaseBitWidth =
      static_cast<uint8_t>(velox::bits::bitsRequired(maxBaseResidual));
  NIMBLE_CHECK_LE(
      exactBaseBitWidth,
      selectedBaseBitWidth,
      "Pfor exact bit width should not exceed selected bit width.");
  const uint8_t baseBitWidth = options.fixedBitWidthUseExactBits
      ? exactBaseBitWidth
      : selectedBaseBitWidth;

  const uint64_t bitpackedSize =
      baseBitWidth == 0 ? 0 : FixedBitArray::bufferSize(rowCount, baseBitWidth);

  // PFOR encodes the exception side-channels through recursive encoding
  // selection so Nimble can pick the best sub-encoding.
  // The bulk base residuals always stay raw to preserve the fast decode path.
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  std::string_view exceptionPositionsEncoded{};
  std::string_view exceptionValuesEncoded{};
  if (numExceptions > 0) {
    exceptionPositionsEncoded = selection.template encodeNested<uint32_t>(
        EncodingIdentifiers::Pfor::ExceptionPositions,
        std::span<const uint32_t>(exceptionPositions.data(), numExceptions),
        scopedBuffer.get(),
        options);
    exceptionValuesEncoded = selection.template encodeNested<physicalType>(
        EncodingIdentifiers::Pfor::ExceptionValues,
        std::span<const physicalType>(exceptionValues.data(), numExceptions),
        scopedBuffer.get(),
        options);
  }
  const auto exceptionPositionsEncodedSize =
      static_cast<uint32_t>(exceptionPositionsEncoded.size());
  const auto exceptionValuesEncodedSize =
      static_cast<uint32_t>(exceptionValuesEncoded.size());
  const uint32_t encodingSize =
      Encoding::serializePrefixSize(rowCount, useVarint) +
      PFOREncoding<T>::kFixedHeaderSize + varint::varintSize(numExceptions) +
      varint::varintSize(exceptionPositionsEncodedSize) +
      varint::varintSize(exceptionValuesEncodedSize) +
      exceptionPositionsEncodedSize + exceptionValuesEncodedSize +
      static_cast<uint32_t>(bitpackedSize);
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::PFOR, TypeTraits<T>::dataType, rowCount, useVarint, pos);
  encoding::write(baseline, pos);
  encoding::writeChar(static_cast<char>(baseBitWidth), pos);
  varint::writeVarint(numExceptions, &pos);
  encoding::writeVarintString(exceptionPositionsEncoded, pos);
  encoding::writeVarintString(exceptionValuesEncoded, pos);
  if (baseBitWidth > 0) {
    std::memset(pos, 0, bitpackedSize);
    FixedBitArray fba(pos, baseBitWidth);
    fba.bulkSetWithBaseline(
        /*start=*/0,
        /*length=*/rowCount,
        maskedResiduals.data(),
        /*baseline=*/physicalType{0});
    pos += bitpackedSize;
  }
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Pfor encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
typename PFOREncoding<T>::ExceptionSlice PFOREncoding<T>::sliceExceptions(
    std::string_view exceptionPositions,
    std::string_view exceptionValues,
    uint32_t numExceptions,
    uint32_t offset,
    uint32_t length,
    velox::memory::MemoryPool* pool,
    Buffer& scratchBuffer,
    const Encoding::Options& options) {
  if (numExceptions == 0) {
    return {};
  }

  NIMBLE_CHECK_NOT_NULL(pool);
  ScopedVector<uint32_t> positions{numExceptions, pool, options.bufferPool};
  auto positionsEncoding =
      EncodingFactory(options).create(*pool, exceptionPositions, nullptr);
  positionsEncoding->materialize(numExceptions, positions.data());

  const auto sliceBegin =
      std::lower_bound(positions.begin(), positions.end(), offset);
  const auto sliceEnd =
      std::lower_bound(sliceBegin, positions.end(), offset + length);
  const auto exceptionOffset =
      static_cast<uint32_t>(sliceBegin - positions.begin());
  const auto slicedExceptionCount =
      static_cast<uint32_t>(sliceEnd - sliceBegin);
  if (slicedExceptionCount == 0) {
    return {};
  }

  ScopedVector<uint32_t> rebasedPositions{
      slicedExceptionCount, pool, options.bufferPool};
  for (uint32_t i = 0; i < slicedExceptionCount; ++i) {
    rebasedPositions[i] = sliceBegin[i] - offset;
  }

  return {
      .positions = EncodingFactory::encodeWithCapturedLayout<uint32_t>(
          exceptionPositions,
          {rebasedPositions.data(), rebasedPositions.size()},
          scratchBuffer,
          options,
          "Captured PFOR exception positions layout"),
      .values = EncodingFactory::slice(
          exceptionValues,
          exceptionOffset,
          slicedExceptionCount,
          scratchBuffer,
          options),
      .count = slicedExceptionCount};
}

template <typename T>
std::string_view PFOREncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  auto* pool = &buffer.getMemoryPool();
  const char* readPos = encoded.data() +
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const auto baseline = encoding::read<physicalType>(readPos);
  const auto baseBitWidth = static_cast<uint8_t>(encoding::readChar(readPos));
  NIMBLE_CHECK_LE(baseBitWidth, 64, "Pfor base bit width must be in [0, 64].");
  const auto numExceptions = varint::readVarint32(&readPos);
  NIMBLE_CHECK_LE(
      numExceptions, sourceRowCount, "Pfor exception count exceeds row count.");
  const auto exceptionPositionsSize = varint::readVarint32(&readPos);
  NIMBLE_CHECK_EQ(
      numExceptions == 0,
      exceptionPositionsSize == 0,
      "Pfor exception positions stream size must match exception count.");
  const std::string_view exceptionPositions{readPos, exceptionPositionsSize};
  readPos += exceptionPositionsSize;
  const auto exceptionValuesSize = varint::readVarint32(&readPos);
  NIMBLE_CHECK_EQ(
      numExceptions == 0,
      exceptionValuesSize == 0,
      "Pfor exception values stream size must match exception count.");
  const std::string_view exceptionValues{readPos, exceptionValuesSize};
  readPos += exceptionValuesSize;
  const std::string_view packedData{
      readPos, static_cast<size_t>(encoded.end() - readPos)};

  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  const auto exceptionSlice = sliceExceptions(
      exceptionPositions,
      exceptionValues,
      numExceptions,
      offset,
      length,
      pool,
      scopedBuffer.get(),
      options);

  const uint32_t bitpackedSize = baseBitWidth == 0
      ? 0
      : static_cast<uint32_t>(FixedBitArray::bufferSize(length, baseBitWidth));
  const auto slicedExceptionPositionsSize =
      static_cast<uint32_t>(exceptionSlice.positions.size());
  const auto slicedExceptionValuesSize =
      static_cast<uint32_t>(exceptionSlice.values.size());
  const uint32_t encodingSize =
      Encoding::serializePrefixSize(length, options.useVarintRowCount) +
      kFixedHeaderSize + varint::varintSize(exceptionSlice.count) +
      varint::varintSize(slicedExceptionPositionsSize) +
      varint::varintSize(slicedExceptionValuesSize) +
      slicedExceptionPositionsSize + slicedExceptionValuesSize + bitpackedSize;
  char* reserved = buffer.reserve(encodingSize);
  char* writePos = reserved;
  Encoding::serializePrefix(
      EncodingType::PFOR,
      TypeTraits<T>::dataType,
      length,
      options.useVarintRowCount,
      writePos);
  encoding::write(baseline, writePos);
  encoding::writeChar(static_cast<char>(baseBitWidth), writePos);
  varint::writeVarint(exceptionSlice.count, &writePos);
  encoding::writeVarintString(exceptionSlice.positions, writePos);
  encoding::writeVarintString(exceptionSlice.values, writePos);
  if (baseBitWidth > 0) {
    std::memset(writePos, 0, bitpackedSize);
    const auto sourceBitOffset = static_cast<uint64_t>(offset) * baseBitWidth;
    const auto sliceBits = static_cast<uint64_t>(length) * baseBitWidth;
    encoding::copyPackedBits(packedData, sourceBitOffset, sliceBits, writePos);
    writePos += bitpackedSize;
  }
  NIMBLE_CHECK_EQ(
      writePos - reserved, encodingSize, "Pfor encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string PFOREncoding<T>::debugString(int offset) const {
  return fmt::format(
      "{}{}<{}> rowCount={} baseBitWidth={} numExceptions={}",
      std::string(offset, ' '),
      toString(Encoding::encodingType()),
      toString(Encoding::dataType()),
      Encoding::rowCount(),
      static_cast<int>(baseBitWidth_),
      numExceptions_);
}

} // namespace facebook::nimble
