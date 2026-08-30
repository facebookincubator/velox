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

#include <xsimd/xsimd.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <vector>
#include "velox/common/encode/Coding.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

/// ALP (Adaptive Lossless floating-Point) encoding for float/double.
///
///
/// Data layout after the standard Encoding prefix:
///   3 bytes: ALP control word:
///     bits 0..4: exponent
///     bits 5..9: factor
///     bits 10..15: reserved
///     bit 16: has exceptions
///     bits 17..23: reserved
///   1-5 bytes: exceptionCount (varint, present only when has exceptions)
///   1-5 bytes: encodedValuesSize (varint uint32, size of nested encoding)
///   N bytes: nested encoding of ZigZag-coded signed encoded values
///   If has exceptions:
///     1-5 bytes: exceptionPositionsSize (varint uint32)
///     N bytes: nested encoding of uint32 exception positions
///     1-5 bytes: exceptionValuesSize (varint uint32)
///     N bytes: nested encoding of original logical exception values

namespace facebook::nimble {

namespace detail::alp {

template <typename FloatType>
inline FloatType toLogical(
    typename TypeTraits<FloatType>::physicalType physicalValue) {
  return EncodingPhysicalType<FloatType>::asEncodingLogicalType(physicalValue);
}

template <typename FloatType>
inline typename TypeTraits<FloatType>::physicalType toPhysical(
    FloatType logicalValue) {
  return EncodingPhysicalType<FloatType>::asEncodingPhysicalType(logicalValue);
}

struct Header {
  // Parameters used to transform floating-point values into integers.
  uint8_t exponent{0};
  uint8_t factor{0};

  // Whether the optional exception count and exception streams are present.
  bool hasExceptions{false};
};

inline Header readHeader(const char*& pos) {
  const auto control = static_cast<uint32_t>(
      static_cast<uint8_t>(pos[0]) | (static_cast<uint8_t>(pos[1]) << 8) |
      (static_cast<uint8_t>(pos[2]) << 16));
  pos += 3;
  return Header{
      .exponent = static_cast<uint8_t>(control & 0x1f),
      .factor = static_cast<uint8_t>((control >> 5) & 0x1f),
      .hasExceptions = (control & (1u << 16)) != 0};
}

inline void writeHeader(const Header& header, char*& pos) {
  NIMBLE_DCHECK_LE(header.exponent, 31);
  NIMBLE_DCHECK_LE(header.factor, 31);
  uint32_t control = header.exponent | (header.factor << 5);
  if (header.hasExceptions) {
    control |= 1u << 16;
  }
  pos[0] = static_cast<char>(control & 0xff);
  pos[1] = static_cast<char>((control >> 8) & 0xff);
  pos[2] = static_cast<char>((control >> 16) & 0xff);
  pos += 3;
}

} // namespace detail::alp

template <typename T>
class ALPEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
  static_assert(
      isFloatingPointType<T>(),
      "ALPEncoding only supports float and double types.");

 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  ALPEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> /* stringBufferFactory */,
      const Encoding::Options& options = {})
      : TypedEncoding<T, physicalType>(pool, data, options), pos_(0) {
    const char* pos = data.data() + this->dataOffset();

    const auto header = detail::alp::readHeader(pos);
    exponent_ = header.exponent;
    factor_ = header.factor;
    exceptionCount_ = header.hasExceptions ? varint::readVarint32(&pos) : 0;
    const uint32_t encodedValuesSize = varint::readVarint32(&pos);

    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    encodedValuesEncoding_ = EncodingFactory().create(
        *this->pool_,
        std::string_view(pos, encodedValuesSize),
        noStringBufferFactory,
        options);
    pos += encodedValuesSize;

    encodedBuffer_.resize(this->rowCount());
    encodedValuesEncoding_->materialize(
        this->rowCount(), encodedBuffer_.data());

    if (exceptionCount_ > 0) {
      const uint32_t exceptionPositionsSize = varint::readVarint32(&pos);
      auto exceptionPositionsEncoding = EncodingFactory().create(
          *this->pool_,
          std::string_view(pos, exceptionPositionsSize),
          noStringBufferFactory,
          options);
      pos += exceptionPositionsSize;

      const uint32_t exceptionValuesSize = varint::readVarint32(&pos);
      auto exceptionValuesEncoding = EncodingFactory().create(
          *this->pool_,
          std::string_view(pos, exceptionValuesSize),
          noStringBufferFactory,
          options);
      pos += exceptionValuesSize;

      exceptionPositionsBuffer_.resize(exceptionCount_);
      exceptionPositionsEncoding->materialize(
          exceptionCount_, exceptionPositionsBuffer_.data());

      exceptionValuesBuffer_.resize(exceptionCount_);
      exceptionValuesEncoding->materialize(
          exceptionCount_, exceptionValuesBuffer_.data());
    }
  }

  void reset() final {
    pos_ = 0;
  }

  void skip(uint32_t rowCount) final {
    pos_ += rowCount;
  }

  void materialize(uint32_t rowCount, void* buffer) final {
    auto* output = static_cast<physicalType*>(buffer);
    for (uint32_t i = 0; i < rowCount; ++i) {
      const auto encoded = velox::ZigZag::decode(encodedBuffer_[pos_ + i]);
      output[i] = detail::alp::toPhysical<cppDataType>(
          decodeValue(encoded, exponent_, factor_));
    }

    patchExceptions(pos_, rowCount, output);
    pos_ += rowCount;
  }

  void materializeBoolsAsBits(
      uint32_t /*rowCount*/,
      uint64_t* /*buffer*/,
      int /*begin*/) final {
    NIMBLE_UNREACHABLE("ALP encoding does not support bool type.");
  }

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params) {
    auto skipFn = [&](auto toSkip) { pos_ += toSkip; };
    auto decodeFn = [&] {
      physicalType value = detail::alp::toPhysical<cppDataType>(decodeValue(
          velox::ZigZag::decode(encodedBuffer_[pos_]), exponent_, factor_));
      if (const auto* exceptionValue = findException(pos_)) {
        value = *exceptionValue;
      }
      ++pos_;
      return value;
    };
    detail::readWithVisitorSlow(visitor, params, skipFn, decodeFn);
  }

  std::string debugString(int offset) const final {
    return fmt::format(
        "{}{}<{}> rowCount={} exponent={} factor={} exceptions={}",
        std::string(offset, ' '),
        this->encodingType(),
        this->dataType(),
        this->rowCount(),
        exponent_,
        factor_,
        exceptionCount_);
  }

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    if (values.empty()) {
      NIMBLE_INCOMPATIBLE_ENCODING("ALP encoding cannot encode empty data.");
    }

    auto* pool = &buffer.getMemoryPool();

    ScopedVector<cppDataType> logicalValues{
        values.size(), pool, options.bufferPool};
    for (uint32_t i = 0; i < values.size(); ++i) {
      logicalValues[i] = detail::alp::toLogical<cppDataType>(values[i]);
    }

    const auto [exponent, factor] = findBestExponentFactorByCount(
        std::span<const cppDataType>{
            logicalValues.data(), logicalValues.size()});

    return encodeWithExponentFactor(
        selection,
        values,
        std::span<const cppDataType>{
            logicalValues.data(), logicalValues.size()},
        exponent,
        factor,
        buffer,
        options);
  }

  // Encodes with a caller-supplied (exponent, factor). Split out from encode()
  // so tests can drive an arbitrary combination end-to-end (e.g. to compare the
  // actual encoded size of count-based vs size-based selection).
  static std::string_view encodeWithExponentFactor(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      std::span<const cppDataType> logicalValues,
      uint8_t exponent,
      uint8_t factor,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    const uint32_t rowCount = values.size();
    auto* pool = &buffer.getMemoryPool();

    ScopedVector<uint64_t> encodedValues{rowCount, pool, options.bufferPool};
    ScopedVector<uint32_t> exceptionPositions{
        /*size=*/0, pool, options.bufferPool};
    ScopedVector<physicalType> exceptionValues{
        /*size=*/0, pool, options.bufferPool};

    // Track the first exactly-representable value's ZigZag encoding so that
    // exception slots can be back-filled with it rather than 0. Writing 0 into
    // exception slots would pollute the frame-of-reference min (and thus the
    // bit-width) of the nested integer encoding, making the real encoded size
    // diverge from the sample-based estimate (see estimateSizeFromSample) and
    // regressing selection quality. This mirrors DuckDB's ALP, which patches
    // exception positions with the first non-exception value. The placeholder
    // is always overwritten by the true value on decode, so correctness is
    // unaffected regardless of which representable value is chosen.
    bool hasPlaceholder = false;
    uint64_t placeholder = 0;
    const double exponentMultiplier = kPow10Double[exponent];
    const double factorMultiplier = kPow10Double[factor];
    alignas(64) uint64_t zigZagLanes[kBatchSize];
    alignas(64) bool okLanes[kBatchSize];

    uint32_t i = 0;
    for (; i + kBatchSize <= rowCount; i += kBatchSize) {
      batchTransform(
          logicalValues.data() + i,
          values.data() + i,
          exponentMultiplier,
          factorMultiplier,
          zigZagLanes,
          okLanes);
      for (std::size_t k = 0; k < kBatchSize; ++k) {
        const uint32_t pos = i + static_cast<uint32_t>(k);
        if (!okLanes[k]) {
          exceptionPositions.push_back(pos);
          exceptionValues.push_back(values[pos]);
          continue;
        }
        encodedValues[pos] = zigZagLanes[k];
        if (!hasPlaceholder) {
          placeholder = encodedValues[pos];
          hasPlaceholder = true;
        }
      }
    }
    for (; i < rowCount; ++i) {
      uint64_t zz = 0;
      if (!scalarTransformOne(
              logicalValues[i],
              values[i],
              exponentMultiplier,
              factorMultiplier,
              zz)) {
        exceptionPositions.push_back(i);
        exceptionValues.push_back(values[i]);
        continue;
      }
      encodedValues[i] = zz;
      if (!hasPlaceholder) {
        placeholder = encodedValues[i];
        hasPlaceholder = true;
      }
    }

    // Back-fill exception slots with the placeholder. When every value is an
    // exception (no representable value exists) the placeholder stays 0, which
    // is fine: the nested stream is uniform and the exception path dominates.
    for (const uint32_t position : exceptionPositions) {
      encodedValues[position] = placeholder;
    }

    const uint32_t exceptionCount = exceptionPositions.size();

    ScopedEncodingBuffer scopedBuffer{
        &buffer.getMemoryPool(), options.encodingBufferPool};
    std::string_view serializedEncoded =
        selection.template encodeNested<uint64_t>(
            EncodingIdentifiers::ALP::EncodedValues,
            {encodedValues.data(), encodedValues.size()},
            scopedBuffer.get(),
            options);

    std::string_view serializedExceptionPositions;
    std::string_view serializedExceptionValues;
    if (exceptionCount > 0) {
      serializedExceptionPositions = selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::ALP::ExceptionPositions,
          {exceptionPositions.data(), exceptionPositions.size()},
          scopedBuffer.get(),
          options);
      serializedExceptionValues = selection.template encodeNested<physicalType>(
          EncodingIdentifiers::ALP::ExceptionValues,
          {exceptionValues.data(), exceptionValues.size()},
          scopedBuffer.get(),
          options);
    }

    const uint32_t metadataSize = kHeaderSize +
        (exceptionCount > 0 ? varint::varintSize(exceptionCount) : 0) +
        varint::varintSize(serializedEncoded.size()) +
        (exceptionCount > 0
             ? varint::varintSize(serializedExceptionPositions.size()) +
                 varint::varintSize(serializedExceptionValues.size())
             : 0);
    const uint32_t encodingSize =
        Encoding::serializePrefixSize(rowCount, options.useVarintRowCount) +
        metadataSize + serializedEncoded.size() +
        serializedExceptionPositions.size() + serializedExceptionValues.size();

    char* reserved = buffer.reserve(encodingSize);
    char* pos = reserved;
    Encoding::serializePrefix(
        EncodingType::ALP,
        TypeTraits<T>::dataType,
        rowCount,
        options.useVarintRowCount,
        pos);

    detail::alp::writeHeader(
        detail::alp::Header{
            .exponent = exponent,
            .factor = factor,
            .hasExceptions = exceptionCount > 0},
        pos);
    if (exceptionCount > 0) {
      varint::writeVarint(exceptionCount, &pos);
    }
    varint::writeVarint(serializedEncoded.size(), &pos);
    encoding::writeBytes(serializedEncoded, pos);

    if (exceptionCount > 0) {
      varint::writeVarint(serializedExceptionPositions.size(), &pos);
      encoding::writeBytes(serializedExceptionPositions, pos);
      varint::writeVarint(serializedExceptionValues.size(), &pos);
      encoding::writeBytes(serializedExceptionValues, pos);
    }

    NIMBLE_CHECK_EQ(encodingSize, pos - reserved, "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    const auto sourceRowCount =
        EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
    NIMBLE_CHECK_LE(offset, sourceRowCount);
    NIMBLE_CHECK_LE(length, sourceRowCount - offset);
    NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

    const char* pos = encoded.data() +
        EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
    const auto header = detail::alp::readHeader(pos);
    const uint32_t exceptionCount =
        header.hasExceptions ? varint::readVarint32(&pos) : 0;
    const uint32_t encodedValuesSize = varint::readVarint32(&pos);
    const std::string_view encodedValues{pos, encodedValuesSize};
    pos += encodedValuesSize;

    auto* pool = &buffer.getMemoryPool();
    ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
    const auto slicedEncodedValues = EncodingFactory::slice(
        encodedValues, offset, length, scopedBuffer.get(), options);
    NIMBLE_CHECK_LE(
        slicedEncodedValues.size(),
        std::numeric_limits<uint32_t>::max(),
        "ALP sliced encoded values are too large.");
    const auto slicedEncodedValuesSize =
        static_cast<uint32_t>(slicedEncodedValues.size());

    const auto slicedExceptionStreams = sliceExceptionStreams(
        pos,
        exceptionCount,
        offset,
        length,
        buffer,
        scopedBuffer.get(),
        options);

    const uint32_t metadataSize = kHeaderSize +
        (slicedExceptionStreams.count > 0
             ? varint::varintSize(slicedExceptionStreams.count)
             : 0) +
        varint::varintSize(slicedEncodedValuesSize) +
        (slicedExceptionStreams.count > 0
             ? varint::varintSize(slicedExceptionStreams.positions.size()) +
                 varint::varintSize(slicedExceptionStreams.values.size())
             : 0);
    const uint32_t encodingSize =
        Encoding::serializePrefixSize(length, options.useVarintRowCount) +
        metadataSize + slicedEncodedValuesSize +
        slicedExceptionStreams.positions.size() +
        slicedExceptionStreams.values.size();

    char* reserved = buffer.reserve(encodingSize);
    char* writePos = reserved;
    Encoding::serializePrefix(
        EncodingType::ALP,
        TypeTraits<T>::dataType,
        length,
        options.useVarintRowCount,
        writePos);
    detail::alp::writeHeader(
        detail::alp::Header{
            .exponent = header.exponent,
            .factor = header.factor,
            .hasExceptions = slicedExceptionStreams.count > 0},
        writePos);
    if (slicedExceptionStreams.count > 0) {
      varint::writeVarint(slicedExceptionStreams.count, &writePos);
    }
    varint::writeVarint(slicedEncodedValuesSize, &writePos);
    encoding::writeBytes(slicedEncodedValues, writePos);
    if (slicedExceptionStreams.count > 0) {
      varint::writeVarint(slicedExceptionStreams.positions.size(), &writePos);
      encoding::writeBytes(slicedExceptionStreams.positions, writePos);
      varint::writeVarint(slicedExceptionStreams.values.size(), &writePos);
      encoding::writeBytes(slicedExceptionStreams.values, writePos);
    }

    NIMBLE_CHECK_EQ(
        encodingSize, writePos - reserved, "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Encoding::Options& options = {}) {
    if (values.empty()) {
      return std::nullopt;
    }

    const uint64_t rowCount = values.size();
    const uint32_t sampleSize = estimateSampleSize(rowCount);

    std::vector<physicalType> sampledValues;
    sampledValues.reserve(sampleSize);
    // Chunked-stride sampling: draw kSamplingChunks equidistant contiguous
    // runs from the input, each of length chunkSize. Contiguous reads let
    // the prefetcher keep up (~32 cache-line pulls instead of ~1024 spread
    // singletons for the default kSampleSize=1024).
    if (rowCount <= kSamplingChunks || sampleSize <= kSamplingChunks) {
      // Fallback for tiny inputs: fall back to strided singleton mapping so
      // small-sample tests keep the exact same rows they did before.
      for (uint32_t i = 0; i < sampleSize; ++i) {
        const auto inputIndex = sampledValueIndex(i, rowCount, sampleSize);
        sampledValues.push_back(values[inputIndex]);
      }
    } else {
      const uint32_t chunkSize = sampleSize / kSamplingChunks;
      for (uint32_t chunk = 0; chunk < kSamplingChunks; ++chunk) {
        const uint64_t chunkStart =
            static_cast<uint64_t>(chunk) * rowCount / kSamplingChunks;
        // Trim the final chunk if it would run past the end of the input;
        // sampledValueIndex clamps the same way when mapping back for
        // exception positions.
        const uint64_t chunkLen =
            std::min<uint64_t>(chunkSize, rowCount - chunkStart);
        const physicalType* p = values.data() + chunkStart;
        for (uint64_t j = 0; j < chunkLen; ++j) {
          sampledValues.push_back(p[j]);
        }
      }
      // If integer division dropped a few slots (sampleSize %
      // kSamplingChunks != 0) top the sample up from the last chunk end so
      // downstream code that assumes sampledValues.size() == sampleSize
      // keeps holding.
      while (sampledValues.size() < sampleSize) {
        const auto inputIndex = sampledValueIndex(
            static_cast<uint32_t>(sampledValues.size()), rowCount, sampleSize);
        sampledValues.push_back(values[inputIndex]);
      }
    }

    return estimateSizeFromSample(rowCount, sampledValues, options);
  }

  static std::optional<uint64_t> estimateSizeFromSample(
      uint64_t rowCount,
      std::span<const physicalType> sampledValues,
      const Encoding::Options& options = {}) {
    NIMBLE_CHECK_GT(rowCount, 0, "ALP estimation requires non-empty input.");
    NIMBLE_CHECK(
        !sampledValues.empty(), "ALP estimation requires a non-empty sample.");
    NIMBLE_CHECK_LE(
        sampledValues.size(),
        rowCount,
        "ALP sample size cannot exceed the input row count.");

    const uint64_t sampleSize = sampledValues.size();

    std::vector<cppDataType> logicalValues;
    logicalValues.reserve(sampleSize);
    for (const auto value : sampledValues) {
      logicalValues.push_back(detail::alp::toLogical<cppDataType>(value));
    }
    const std::span<const cppDataType> logicalSpan{
        logicalValues.data(), logicalValues.size()};

    // Pick (exponent, factor) with the shared selector, then re-score the
    // winner over the same sample to recover its ZigZag min/max and exception
    // count.  Re-scoring costs one extra pass out of the O(kMaxE * kMaxF) grid
    // the selector already visited -- negligible -- and guarantees the
    // integer-stream cost model here is byte-identical to what the selector
    // saw when it chose (exponent, factor).  This eliminates the earlier
    // duplicate "encodedValues + Statistics<uint64_t>" scan that was subtly
    // inconsistent with the selector's min/max view.
    const auto [exponent, factor] = findBestExponentFactorByCount(logicalSpan);
    const uint32_t scoreSampleSize =
        std::min<uint32_t>(static_cast<uint32_t>(sampleSize), kSampleSize);
    const auto winnerScore = scoreCombination(
        logicalSpan.subspan(0, scoreSampleSize), exponent, factor, options);

    // Collect exception positions/values so their per-stream FBW/Trivial cost
    // is exact.  Sized to the known exception count from scoreCombination --
    // no over-reservation -- but iterating the sample once is unavoidable
    // because the exception *values* and their *input positions* (via
    // sampledValueIndex) are what those two nested streams encode.
    std::vector<uint32_t> exceptionPositions;
    std::vector<physicalType> exceptionValues;
    exceptionPositions.reserve(winnerScore.exceptionCount);
    exceptionValues.reserve(winnerScore.exceptionCount);
    uint64_t sampleExceptionCount{0};
    for (uint64_t i = 0; i < sampleSize; ++i) {
      if (!canRepresentExactly(
              logicalValues[i],
              sampledValues[static_cast<size_t>(i)],
              exponent,
              factor)) {
        exceptionPositions.push_back(
            static_cast<uint32_t>(sampledValueIndex(
                static_cast<uint32_t>(i),
                rowCount,
                static_cast<uint32_t>(sampleSize))));
        exceptionValues.push_back(sampledValues[static_cast<size_t>(i)]);
        ++sampleExceptionCount;
      }
    }

    // Integer stream: use the selector's ZigZag range directly.  Because
    // encodeWithExponentFactor back-fills exception slots with the first
    // representable value's ZigZag encoding (never 0), the min/max over the
    // full encoded stream equal the min/max over the representable subset --
    // exactly what scoreCombination computed above.  When every sampled value
    // is an exception, winnerScore.estimatedBytes == kUnusableScore and both
    // min/max come back as 0, producing a zero-range integer stream sized to
    // rowCount -- same behavior as the pre-unification code path.
    const uint64_t nestedEncodedValuesSize = std::min(
        FixedBitWidthEncoding<uint64_t>::estimateSize(
            rowCount, winnerScore.zigZagMin, winnerScore.zigZagMax, options),
        TrivialEncoding<uint64_t>::estimateSize(rowCount));
    const uint64_t exceptionCount =
        (sampleExceptionCount * rowCount + sampleSize - 1) / sampleSize;
    uint64_t exceptionPositionsSize{0};
    uint64_t exceptionValuesSize{0};
    if (exceptionCount > 0) {
      const auto positionStats = Statistics<uint32_t>::create(
          std::span<const uint32_t>{
              exceptionPositions.data(), exceptionPositions.size()});
      exceptionPositionsSize = std::min(
          TrivialEncoding<uint32_t>::estimateSize(exceptionCount),
          FixedBitWidthEncoding<uint32_t>::estimateSize(
              exceptionCount, positionStats, options));

      const auto valueStats = Statistics<physicalType>::create(
          std::span<const physicalType>{
              exceptionValues.data(), exceptionValues.size()});
      exceptionValuesSize = std::min(
          TrivialEncoding<physicalType>::estimateSize(exceptionCount),
          FixedBitWidthEncoding<physicalType>::estimateSize(
              exceptionCount, valueStats, options));
    }
    const uint64_t metadataSize = kHeaderSize +
        (exceptionCount > 0 ? varint::varintSize(exceptionCount) : 0) +
        varint::varintSize(nestedEncodedValuesSize) +
        (exceptionCount > 0 ? varint::varintSize(exceptionPositionsSize) +
                 varint::varintSize(exceptionValuesSize)
                            : 0);
    return Encoding::serializePrefixSize(
               static_cast<uint32_t>(rowCount), options.useVarintRowCount) +
        metadataSize + nestedEncodedValuesSize + exceptionPositionsSize +
        exceptionValuesSize;
  }

  static uint32_t estimateSampleSize(uint64_t rowCount) {
    return std::min(static_cast<uint32_t>(rowCount), kSampleSize);
  }

  /// Maps a dense sample ordinal (0..sampleSize-1) to an input row using a
  /// chunked-stride layout: the sample is split into kSamplingChunks
  /// equidistant contiguous chunks that together cover the input span. This
  /// is cheaper than picking sampleSize far-apart singletons because each
  /// chunk touches only a handful of cache lines. When the input is smaller
  /// than kSamplingChunks the layout collapses to a single dense chunk
  /// (rowCount rows, chunkSize == sampleSize == rowCount), matching the old
  /// behavior for small inputs so unit tests that pin (e, f) on tiny samples
  /// stay stable.
  static uint64_t sampledValueIndex(
      uint32_t sampleIndex,
      uint64_t rowCount,
      uint32_t sampleSize) {
    if (sampleSize == 0) {
      return 0;
    }
    // A single dense chunk when the input is too small to split.
    if (rowCount <= kSamplingChunks || sampleSize <= kSamplingChunks) {
      return static_cast<uint64_t>(sampleIndex) * rowCount / sampleSize;
    }
    const uint32_t chunkSize = sampleSize / kSamplingChunks;
    const uint32_t chunkIdx = sampleIndex / chunkSize;
    const uint32_t offsetInChunk = sampleIndex % chunkSize;
    // Chunk starting positions are evenly spaced across [0, rowCount);
    // rounding is toward 0 so the final chunk still fits when
    // rowCount is not an exact multiple of kSamplingChunks.
    const uint64_t chunkStart =
        static_cast<uint64_t>(chunkIdx) * rowCount / kSamplingChunks;
    // Clamp: the last chunk may abut rowCount if chunkSize > residual;
    // clamping to (rowCount - 1) preserves the invariant that every
    // returned index is in-range.
    const uint64_t idx = chunkStart + offsetInChunk;
    return idx < rowCount ? idx : rowCount - 1;
  }

 private:
  struct SlicedExceptionStreams {
    uint32_t count{0};
    std::string_view positions;
    std::string_view values;
  };

  static SlicedExceptionStreams sliceExceptionStreams(
      const char*& pos,
      uint32_t exceptionCount,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      Buffer& scopedBuffer,
      const Encoding::Options& options) {
    if (exceptionCount == 0) {
      return {};
    }

    auto* pool = &buffer.getMemoryPool();
    ScopedVector<uint32_t> exceptionPositions{
        exceptionCount, pool, options.bufferPool};

    const uint32_t exceptionPositionsSize = varint::readVarint32(&pos);
    const std::string_view exceptionPositionsEncoded{
        pos, exceptionPositionsSize};
    auto exceptionPositionsEncoding = EncodingFactory(options).create(
        buffer.getMemoryPool(),
        exceptionPositionsEncoded,
        [](uint32_t) -> void* { return nullptr; });
    pos += exceptionPositionsSize;
    exceptionPositionsEncoding->materialize(
        exceptionCount, exceptionPositions.data());

    const uint32_t exceptionValuesSize = varint::readVarint32(&pos);
    const std::string_view exceptionValuesEncoded{pos, exceptionValuesSize};
    pos += exceptionValuesSize;

    const auto* exceptionPositionsBegin = exceptionPositions.data();
    const auto* exceptionPositionsEnd =
        exceptionPositionsBegin + exceptionCount;
    const uint32_t firstExceptionIndex = static_cast<uint32_t>(
        std::lower_bound(
            exceptionPositionsBegin, exceptionPositionsEnd, offset) -
        exceptionPositionsBegin);
    const uint32_t lastExceptionIndex = static_cast<uint32_t>(
        std::lower_bound(
            exceptionPositionsBegin + firstExceptionIndex,
            exceptionPositionsEnd,
            offset + length) -
        exceptionPositionsBegin);
    const auto slicedExceptionCount = lastExceptionIndex - firstExceptionIndex;
    if (slicedExceptionCount == 0) {
      return {};
    }

    ScopedVector<uint32_t> slicedExceptionPositions{
        slicedExceptionCount, pool, options.bufferPool};
    uint32_t exceptionIndex{0};
    for (uint32_t i = firstExceptionIndex; i < lastExceptionIndex; ++i) {
      slicedExceptionPositions[exceptionIndex] = exceptionPositions[i] - offset;
      ++exceptionIndex;
    }
    NIMBLE_CHECK_EQ(exceptionIndex, slicedExceptionCount);

    auto slicedExceptionPositionsEncoded =
        EncodingFactory::encodeWithCapturedLayout<uint32_t>(
            exceptionPositionsEncoded,
            {slicedExceptionPositions.data(), slicedExceptionCount},
            scopedBuffer,
            options,
            "ALP exception positions layout");
    NIMBLE_CHECK_LE(
        slicedExceptionPositionsEncoded.size(),
        std::numeric_limits<uint32_t>::max(),
        "ALP sliced exception positions are too large.");

    auto slicedExceptionValuesEncoded = EncodingFactory::slice(
        exceptionValuesEncoded,
        firstExceptionIndex,
        slicedExceptionCount,
        scopedBuffer,
        options);
    NIMBLE_CHECK_LE(
        slicedExceptionValuesEncoded.size(),
        std::numeric_limits<uint32_t>::max(),
        "ALP sliced exception values are too large.");

    return {
        .count = slicedExceptionCount,
        .positions = slicedExceptionPositionsEncoded,
        .values = slicedExceptionValuesEncoded};
  }

 public:
  // Pre-computed powers of 10 for double precision.
  static constexpr std::array<double, 24> kPow10Double{
      1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
      1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23};

  // Sample up to this many values to find the best (exponent, factor) pair.
  static constexpr uint32_t kSampleSize{1024};
  // Number of equidistant chunks the sample is drawn from. Each chunk is a
  // contiguous run of kSampleSize / kSamplingChunks values from the input.
  // Total distinct sample count is still kSampleSize; the difference vs.
  // strided single-value sampling is that we take 32 short contiguous runs
  // rather than 1024 far-apart singletons -- ~32 cache-line pulls instead of
  // ~1024, which is cheaper on the estimate hot path and still covers the
  // input evenly. DuckDB uses the same layout (SAMPLES_PER_VECTOR=32) for the
  // same reason.
  static constexpr uint32_t kSamplingChunks{32};

 private:
  // Largest exponent and factor values backed by kPow10Double.
  static constexpr int kMaxExponent{23};
  static constexpr int kMaxFactor{23};
  // ALP-specific control word following the standard Encoding prefix.
  static constexpr uint32_t kHeaderSize{3};

  // Checks whether the selected ALP transform can encode the value without an
  // exception.
  static bool canRepresentExactly(
      cppDataType value,
      physicalType physicalValue,
      int exponent,
      int factor) {
    const double exponentMultiplier = kPow10Double[exponent];
    const double factorMultiplier = kPow10Double[factor];
    const double scaled = static_cast<double>(value) * exponentMultiplier;
    if (!std::isfinite(scaled)) {
      return false;
    }
    if (scaled < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
        scaled > static_cast<double>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    const int64_t factored =
        static_cast<int64_t>(std::llround(scaled / factorMultiplier));
    const double restored =
        static_cast<double>(factored) * factorMultiplier / exponentMultiplier;

    return detail::alp::toPhysical<cppDataType>(
               static_cast<cppDataType>(restored)) == physicalValue;
  }

 public:
  // Scalar version of the per-row ALP transform. Merges canRepresentExactly
  // and encodeValue into a single pass so scaled/factored/restored are each
  // computed once. Writes the ZigZag-encoded uint64 to `zigZagOut` and returns
  // true iff the value is exactly representable under (exponent, factor).
  // Byte-identical to `canRepresentExactly` + `encodeValue` + `ZigZag::encode`
  // when the return value is true.
  static bool scalarTransformOne(
      cppDataType logical,
      physicalType physical,
      double exponentMultiplier,
      double factorMultiplier,
      uint64_t& zigZagOut) {
    const double scaled = static_cast<double>(logical) * exponentMultiplier;
    if (!std::isfinite(scaled)) {
      return false;
    }
    if (scaled < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
        scaled > static_cast<double>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    const int64_t factored =
        static_cast<int64_t>(std::llround(scaled / factorMultiplier));
    const double restored =
        static_cast<double>(factored) * factorMultiplier / exponentMultiplier;
    if (detail::alp::toPhysical<cppDataType>(
            static_cast<cppDataType>(restored)) != physical) {
      return false;
    }
    zigZagOut = velox::ZigZag::encode(factored);
    return true;
  }

  // Vectorized version of `scalarTransformOne` over a contiguous block. For
  // each lane, computes multiply/round/inverse-transform in xsimd, then falls
  // back to per-lane scalar for the physical-byte equality check (which is a
  // cheap 32- or 64-bit compare on values that are already in registers).
  //
  // The rounding path uses xsimd::trunc(x + copysign(0.5, x)), which is the
  // vectorizable equivalent of std::llround's round-half-away-from-zero rule.
  // A per-lane byte-agreement UT (batchTransformMatchesScalar) locks this in.
  //
  // Writes count lanes starting at `outZigZag` and sets outMask[i]=true iff
  // lane i is exactly representable. Undefined lanes in outZigZag when the
  // corresponding mask is false. `count` must be exactly the SIMD batch size
  // (`kBatchSize`); scalar tails are handled by callers via scalarTransformOne.
  static constexpr std::size_t kBatchSize = xsimd::batch<double>::size;
  using BatchD = xsimd::batch<double>;

  static void batchTransform(
      const cppDataType* logicals,
      const physicalType* physicals,
      double exponentMultiplier,
      double factorMultiplier,
      uint64_t* outZigZag,
      bool* outMask) {
    // Widen to double for float inputs so all lanes share the same rounding
    // domain as std::llround(double).
    alignas(64) double lanes[kBatchSize];
    for (std::size_t i = 0; i < kBatchSize; ++i) {
      lanes[i] = static_cast<double>(logicals[i]);
    }
    const auto x = BatchD::load_aligned(lanes);
    const auto scaled = x * BatchD(exponentMultiplier);
    // Range gate: !isfinite(scaled) OR scaled out of int64 domain -> exception.
    // Building the mask as an integer bit-mask keeps the branch out of the
    // hot lane; the tail scalar path re-computes the same expression.
    const auto finiteMask = xsimd::isfinite(scaled);
    const auto lowBound =
        BatchD(static_cast<double>(std::numeric_limits<int64_t>::min()));
    const auto highBound =
        BatchD(static_cast<double>(std::numeric_limits<int64_t>::max()));
    const auto inRangeMask = (scaled >= lowBound) & (scaled <= highBound);
    const auto safeMask = finiteMask & inRangeMask;

    // Round-half-away-from-zero. `xsimd::round` matches `std::round` (and
    // therefore `std::llround` for finite in-range values). We avoid the
    // `trunc(x + copysign(0.5, x))` emulation because at magnitudes near
    // 2^52..2^53 double ULP == 1.0, so the added 0.5 rounds in FP and
    // diverges from `std::llround` — the byte-agreement UT catches this.
    const auto div = scaled / BatchD(factorMultiplier);
    const auto rounded = xsimd::round(div);
    // Convert back to int64 lane-by-lane for the equality check and ZigZag.
    // Undefined lanes (mask=false) are still safe to convert because their
    // final result is discarded via outMask.
    alignas(64) double roundedLanes[kBatchSize];
    rounded.store_aligned(roundedLanes);

    // Materialize the range/finite mask as per-lane 1.0/0.0 doubles so we
    // can read it back cheaply lane-by-lane. Direct storage of
    // `xsimd::batch_bool<double>` varies by architecture; going through
    // select-to-double keeps the code portable across AVX-2, AVX-512, and
    // NEON.
    alignas(64) double safeLanes[kBatchSize];
    xsimd::select(safeMask, BatchD(1.0), BatchD(0.0)).store_aligned(safeLanes);
    for (std::size_t i = 0; i < kBatchSize; ++i) {
      if (safeLanes[i] == 0.0) {
        outMask[i] = false;
        continue;
      }
      // Restore path must go through int64 -> double, matching scalar exactly.
      // A pure FP-domain restore would preserve -0.0 and any near-boundary
      // rounding artifacts that scalar's int64 cast collapses; the byte-
      // agreement UT catches that divergence, so we mirror scalar here.
      const int64_t factored = static_cast<int64_t>(roundedLanes[i]);
      const double restoredD =
          static_cast<double>(factored) * factorMultiplier / exponentMultiplier;
      const cppDataType restored = static_cast<cppDataType>(restoredD);
      if (detail::alp::toPhysical<cppDataType>(restored) != physicals[i]) {
        outMask[i] = false;
        continue;
      }
      outMask[i] = true;
      outZigZag[i] = velox::ZigZag::encode(factored);
    }
  }

  // Estimated encoded footprint of a single (exponent, factor) candidate over
  // representable values -- lets selection weigh the FOR/bit-packing cost of a
  // larger integer domain against a lower exception rate, matching the ALP
  // paper and DuckDB. This is the single source of truth shared by size-based
  // selection and estimateSizeFromSample.
 public:
  struct CombinationScore {
    // Estimated total bytes for the integer stream plus exception payload.
    // kUnusable when the candidate produces no representable values.
    uint64_t estimatedBytes;
    uint32_t exceptionCount;
    // ZigZag range over the representable-value stream. Undefined when the
    // combination is unusable (representableCount == 0). Reusing these in
    // estimateSizeFromSample avoids a second scan of the sample and keeps the
    // selector's byte estimate byte-identical to the estimator's integer
    // stream sizing.
    uint64_t zigZagMin;
    uint64_t zigZagMax;
  };

  static constexpr uint64_t kUnusableScore =
      std::numeric_limits<uint64_t>::max();

  static CombinationScore scoreCombination(
      std::span<const cppDataType> logicalValues,
      int exponent,
      int factor,
      const Encoding::Options& options) {
    const uint64_t sampleSize = logicalValues.size();
    uint64_t zigZagMin = std::numeric_limits<uint64_t>::max();
    uint64_t zigZagMax = 0;
    uint32_t exceptionCount = 0;
    uint32_t representableCount = 0;

    const double exponentMultiplier = kPow10Double[exponent];
    const double factorMultiplier = kPow10Double[factor];

    // Vectorized main loop: process kBatchSize lanes at a time through
    // batchTransform, then handle the tail with scalarTransformOne. The batch
    // path is byte-identical to the scalar path (locked in by
    // batchTransformMatchesScalar UT), so this only changes throughput, not
    // the ZigZag range or exception count observed by size-based selection.
    alignas(64) uint64_t zigZagLanes[kBatchSize];
    alignas(64) bool okLanes[kBatchSize];
    alignas(64) physicalType physLanes[kBatchSize];

    uint64_t i = 0;
    for (; i + kBatchSize <= sampleSize; i += kBatchSize) {
      for (std::size_t k = 0; k < kBatchSize; ++k) {
        physLanes[k] =
            detail::alp::toPhysical<cppDataType>(logicalValues[i + k]);
      }
      batchTransform(
          logicalValues.data() + i,
          physLanes,
          exponentMultiplier,
          factorMultiplier,
          zigZagLanes,
          okLanes);
      for (std::size_t k = 0; k < kBatchSize; ++k) {
        if (!okLanes[k]) {
          ++exceptionCount;
          continue;
        }
        const uint64_t zz = zigZagLanes[k];
        zigZagMin = std::min(zigZagMin, zz);
        zigZagMax = std::max(zigZagMax, zz);
        ++representableCount;
      }
    }
    for (; i < sampleSize; ++i) {
      const auto logical = logicalValues[i];
      uint64_t zz = 0;
      if (!scalarTransformOne(
              logical,
              detail::alp::toPhysical<cppDataType>(logical),
              exponentMultiplier,
              factorMultiplier,
              zz)) {
        ++exceptionCount;
        continue;
      }
      zigZagMin = std::min(zigZagMin, zz);
      zigZagMax = std::max(zigZagMax, zz);
      ++representableCount;
    }

    if (representableCount == 0) {
      // Every sampled value is an exception; this combination cannot be used.
      return {kUnusableScore, exceptionCount, 0, 0};
    }

    // Bit-packing operates on the ZigZag-encoded integers, so the FOR bit width
    // must be derived from their min/max (mirrors estimateSizeFromSample).
    const uint64_t integerStreamBytes = std::min(
        FixedBitWidthEncoding<uint64_t>::estimateSize(
            sampleSize, zigZagMin, zigZagMax, options),
        TrivialEncoding<uint64_t>::estimateSize(sampleSize));
    const uint64_t exceptionBytes = static_cast<uint64_t>(exceptionCount) *
        (sizeof(uint32_t) + sizeof(physicalType));
    return {
        integerStreamBytes + exceptionBytes,
        exceptionCount,
        zigZagMin,
        zigZagMax};
  }

  // Selects the (exponent, factor) pair with the smallest estimated encoded
  // footprint. Ties prefer the larger exponent, then the larger factor
  // (DuckDB's tie-break rule). Retained for tests, benchmarks, and A/B
  // exploration; production paths use findBestExponentFactorByCount, which is
  // cheaper and matches this on all datasets except sparse-exception ones.
  static std::pair<uint8_t, uint8_t> findBestExponentFactorBySize(
      std::span<const cppDataType> values,
      const Encoding::Options& options) {
    const uint32_t sampleSize =
        std::min(static_cast<uint32_t>(values.size()), kSampleSize);
    const std::span<const cppDataType> sample{values.data(), sampleSize};

    uint8_t bestExponent = 0;
    uint8_t bestFactor = 0;
    uint64_t bestBytes = kUnusableScore;

    for (int e = 0; e <= kMaxExponent; ++e) {
      for (int f = 0; f <= std::min(e, kMaxFactor); ++f) {
        const auto score = scoreCombination(sample, e, f, options);
        if (score.estimatedBytes == kUnusableScore) {
          continue;
        }
        // Strictly smaller wins outright; iterating exponent and factor in
        // ascending order with <= lets the largest (exponent, factor) among
        // equal-cost candidates win, matching DuckDB's tie-break rule (keep
        // more decimal precision so out-of-sample values stay representable).
        if (score.estimatedBytes <= bestBytes) {
          bestBytes = score.estimatedBytes;
          bestExponent = static_cast<uint8_t>(e);
          bestFactor = static_cast<uint8_t>(f);
        }
      }
    }
    return {bestExponent, bestFactor};
  }

  // Selects the sampled (exponent, factor) pair that preserves the most values.
  // Ties break toward the smaller pair, and iteration short-circuits when every
  // sampled value is representable -- clean data (all integers, uniform 2dp)
  // exits on the first candidate.
  static std::pair<uint8_t, uint8_t> findBestExponentFactorByCount(
      std::span<const cppDataType> values) {
    const uint32_t sampleSize =
        std::min(static_cast<uint32_t>(values.size()), kSampleSize);

    uint8_t bestExponent = 0;
    uint8_t bestFactor = 0;
    uint32_t bestRepresentableCount = 0;

    for (int e = 0; e <= kMaxExponent; ++e) {
      uint32_t countNoFactor = 0;
      for (uint32_t i = 0; i < sampleSize; ++i) {
        if (canRepresentExactly(
                values[i],
                detail::alp::toPhysical<cppDataType>(values[i]),
                e,
                /*factor=*/0)) {
          ++countNoFactor;
        }
      }
      if (countNoFactor > bestRepresentableCount) {
        bestRepresentableCount = countNoFactor;
        bestExponent = static_cast<uint8_t>(e);
        bestFactor = 0;
      }
      if (bestRepresentableCount == sampleSize) {
        break;
      }

      for (int f = 1; f <= std::min(e, kMaxFactor); ++f) {
        uint32_t countWithFactor = 0;
        for (uint32_t i = 0; i < sampleSize; ++i) {
          if (canRepresentExactly(
                  values[i],
                  detail::alp::toPhysical<cppDataType>(values[i]),
                  e,
                  f)) {
            ++countWithFactor;
          }
        }
        if (countWithFactor > bestRepresentableCount) {
          bestRepresentableCount = countWithFactor;
          bestExponent = static_cast<uint8_t>(e);
          bestFactor = static_cast<uint8_t>(f);
        }
        if (bestRepresentableCount == sampleSize) {
          break;
        }
      }
      if (bestRepresentableCount == sampleSize) {
        break;
      }
    }
    return {bestExponent, bestFactor};
  }

 private:
  // Converts a floating-point value to the integer stored by ALP.
  static int64_t encodeValue(double value, int exponent, int factor) {
    const double scaled = value * kPow10Double[exponent];
    return static_cast<int64_t>(std::llround(scaled / kPow10Double[factor]));
  }

  // Reconstructs a floating-point value from an ALP integer.
  static cppDataType decodeValue(int64_t encoded, int exponent, int factor) {
    return static_cast<cppDataType>(
        static_cast<double>(encoded) * kPow10Double[factor] /
        kPow10Double[exponent]);
  }

  void
  patchExceptions(uint32_t startRow, uint32_t rowCount, physicalType* output) {
    if (exceptionCount_ == 0) {
      return;
    }
    const auto* exVals = exceptionValuesBuffer_.data();
    const auto* exBegin = exceptionPositionsBuffer_.data();
    const auto* exEnd = exBegin + exceptionCount_;
    const auto* first = std::lower_bound(exBegin, exEnd, startRow);
    const auto* last = std::lower_bound(first, exEnd, startRow + rowCount);
    for (auto* it = first; it != last; ++it) {
      const uint32_t absPos = *it;
      output[absPos - startRow] = exVals[it - exBegin];
    }
  }

  const physicalType* findException(uint32_t row) const {
    if (exceptionCount_ == 0) {
      return nullptr;
    }
    const auto* exVals = exceptionValuesBuffer_.data();
    const auto* exBegin = exceptionPositionsBuffer_.data();
    const auto* exEnd = exBegin + exceptionCount_;
    const auto* it = std::lower_bound(exBegin, exEnd, row);
    if (it == exEnd || *it != row) {
      return nullptr;
    }
    return exVals + (it - exBegin);
  }

  uint8_t exponent_;
  uint8_t factor_;
  uint32_t exceptionCount_;
  std::unique_ptr<Encoding> encodedValuesEncoding_;
  std::vector<uint64_t> encodedBuffer_;
  std::vector<uint32_t> exceptionPositionsBuffer_;
  std::vector<physicalType> exceptionValuesBuffer_;
  uint32_t pos_;
};

} // namespace facebook::nimble
