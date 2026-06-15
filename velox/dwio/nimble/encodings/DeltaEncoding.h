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

#include <numeric>
#include <span>

#include <folly/Likely.h>

#include "velox/buffer/Buffer.h"
#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Stores integer data in a delta encoding. We use three child encodings:
// one for whether each row is a delta from the last or a restatement,
// one for the deltas, and one for the restatements. For now we
// only support positive deltas.
//
// As an example, consider the data
//
// 1 2 4 1 2 3 4 1 2 4 8 8
//
// The is-restatement  bool vector is
// T F F T F F F T F F F F
//
// The delta vector is
// 1 2 1 1 1 1 2 4 0
//
// The restatement vector is
// 1 1 1

namespace facebook::nimble {

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 4 bytes: restatement relative offset (X)
// 4 bytes: is-restatement relative offset (Y)
// X bytes: delta encoding bytes
// Y bytes: restatement encoding bytes
// Z bytes: is-restatement encoding bytes
template <typename T>
class DeltaEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  DeltaEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  ~DeltaEncoding() override {
    this->releaseVectorBuffer(deltasBuffer_);
    this->releaseVectorBuffer(restatementsBuffer_);
    this->releaseBuffer(isRestatementsBitmap_);
  }

  DeltaEncoding(const DeltaEncoding&) = delete;
  DeltaEncoding& operator=(const DeltaEncoding&) = delete;
  DeltaEncoding(DeltaEncoding&&) = delete;
  DeltaEncoding& operator=(DeltaEncoding&&) = delete;

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  std::string debugString(int offset) const final;

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  /// Statistics-only size estimate for general encoding selection (e.g. as a
  /// SubIntSplit segment candidate), where only `Statistics<physicalType>` --
  /// not the raw values -- is available. Without raw values, the true
  /// monotonic-step pattern can't be observed; this assumes a single leading
  /// restatement and an average step size of range / (rowCount - 1) -- the
  /// typical step for a column whose values are roughly evenly spread across
  /// [min, max] in row order. This is a coarse approximation: non-monotonic
  /// columns (which need many restatements) will be underestimated.
  static uint64_t estimateSize(
      uint64_t rowCount,
      const Statistics<physicalType>& statistics) {
    if (rowCount == 0) {
      return EncodingPrefix::kFixedPrefixSize;
    }
    const auto fullRange =
        static_cast<uint64_t>(statistics.max() - statistics.min());
    const double avgAbsDelta = rowCount > 1
        ? static_cast<double>(fullRange) / static_cast<double>(rowCount - 1)
        : 0.0;
    const uint8_t deltaBitWidth = avgAbsDelta < 1.0
        ? uint8_t{0}
        : static_cast<uint8_t>(
              velox::bits::bitsRequired(static_cast<uint64_t>(avgAbsDelta)));

    const uint64_t deltasSize =
        FixedBitArray::bufferSize(rowCount, deltaBitWidth);
    // Assume a single leading restatement (best case; non-monotonic columns
    // need more, but Statistics<T> doesn't expose monotonicity).
    const uint64_t restatementsSize = sizeof(physicalType);
    const uint64_t isRestatementsSize = velox::bits::nbytes(rowCount);

    // Each of the three nested sub-streams has its own ~7-byte header
    // (prefix(6) + compressionType(1)).
    constexpr uint64_t kNestedHeaderSize = 7;
    // Outer prefix(6) + two 4-byte relative offsets.
    constexpr uint64_t kOuterHeaderSize = EncodingPrefix::kFixedPrefixSize + 8;

    return kOuterHeaderSize + (kNestedHeaderSize + deltasSize) +
        (kNestedHeaderSize + restatementsSize) +
        (kNestedHeaderSize + isRestatementsSize);
  }
#endif

 private:
  // Ensures isRestatementsBitmap_ has capacity for rowCount bits and
  // returns a mutable pointer to the underlying uint64_t words.
  uint64_t* ensureRestatementsBitmap(uint32_t rowCount) {
    const auto bitmapBytes = velox::bits::nwords(rowCount) * sizeof(uint64_t);
    if (isRestatementsBitmap_ == nullptr ||
        isRestatementsBitmap_->capacity() < bitmapBytes) {
      isRestatementsBitmap_ = this->getBuffer(bitmapBytes);
    }
    return isRestatementsBitmap_->asMutable<uint64_t>();
  }

  physicalType currentValue_;
  std::unique_ptr<Encoding> deltas_;
  std::unique_ptr<Encoding> restatements_;
  std::unique_ptr<Encoding> isRestatements_;
  // Temporary bufs.
  Vector<physicalType> deltasBuffer_;
  Vector<physicalType> restatementsBuffer_;
  velox::BufferPtr isRestatementsBitmap_;
};

//
// End of public API. Implementation follows.
//

template <typename T>
DeltaEncoding<T>::DeltaEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>(pool, data, options),
      deltasBuffer_(this->template getVectorBuffer<physicalType>()),
      restatementsBuffer_(this->template getVectorBuffer<physicalType>()) {
  const EncodingFactory factory{options};
  auto pos = data.data() + this->dataOffset();
  const uint32_t restatementsOffset = encoding::readUint32(pos);
  const uint32_t isRestatementsOffset = encoding::readUint32(pos);
  deltas_ =
      factory.create(pool, {pos, restatementsOffset}, stringBufferFactory);
  pos += restatementsOffset;
  restatements_ =
      factory.create(pool, {pos, isRestatementsOffset}, stringBufferFactory);
  pos += isRestatementsOffset;
  isRestatements_ = factory.create(
      pool,
      {pos, static_cast<size_t>(data.end() - pos)},
      std::move(stringBufferFactory));
}

template <typename T>
void DeltaEncoding<T>::reset() {
  deltas_->reset();
  restatements_->reset();
  isRestatements_->reset();
}

template <typename T>
void DeltaEncoding<T>::skip(uint32_t rowCount) {
  if (rowCount == 0) {
    return;
  }

  auto* bitmap = ensureRestatementsBitmap(rowCount);
  isRestatements_->materializeBoolsAsBits(rowCount, bitmap, 0);

  const uint32_t totalRestatements =
      velox::bits::countBits(bitmap, 0, rowCount);

  // Find the last restatement position using reverse bit scan.
  int64_t lastRestatement = -1;
  if (totalRestatements > 0) {
    lastRestatement = velox::bits::findLastBit(bitmap, 0, rowCount);
  }

  if (lastRestatement >= 0) {
    restatements_->skip(totalRestatements - 1);
    restatements_->materialize(1, &currentValue_);
    const uint32_t deltasToSkip =
        static_cast<uint32_t>(lastRestatement) - (totalRestatements - 1);
    deltas_->skip(deltasToSkip);
  }
  const uint32_t deltasToAccumulate =
      static_cast<uint32_t>(rowCount - 1 - lastRestatement);
  deltasBuffer_.resize(deltasToAccumulate);
  deltas_->materialize(deltasToAccumulate, deltasBuffer_.data());
  currentValue_ += std::accumulate(
      deltasBuffer_.begin(), deltasBuffer_.end(), physicalType());
}

template <typename T>
void DeltaEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  // Decode isRestatements as bit-packed bitmap and count via popcount.
  auto* bitmap = ensureRestatementsBitmap(rowCount);
  isRestatements_->materializeBoolsAsBits(rowCount, bitmap, 0);

  const uint32_t numRestatements = velox::bits::countBits(bitmap, 0, rowCount);

  restatementsBuffer_.reserve(numRestatements);
  restatements_->materialize(numRestatements, restatementsBuffer_.data());
  deltasBuffer_.reserve(rowCount - numRestatements);
  deltas_->materialize(rowCount - numRestatements, deltasBuffer_.data());

  auto* output = static_cast<physicalType*>(buffer);
  const auto* nextRestatement = restatementsBuffer_.data();
  const auto* nextDelta = deltasBuffer_.data();

  // Process the restatement bitmap 16 bits at a time. For all-delta
  // chunks (the common case in sorted data), runs a tight branchless
  // prefix-sum loop without per-element bit extraction or branching.
  // 16-bit chunks balance fast-path hit rate with loop overhead
  // (benchmarked vs 8/32/64-bit: 16-bit is fastest).
  uint32_t remaining = rowCount;
  const auto* bitmapChunks = reinterpret_cast<const uint16_t*>(bitmap);
  const uint32_t numChunks = velox::bits::divRoundUp(rowCount, 16);
  for (uint32_t c = 0; c < numChunks; ++c) {
    const uint16_t chunk = bitmapChunks[c];
    const uint32_t count = std::min<uint32_t>(16, remaining);

    if (FOLLY_LIKELY(chunk == 0)) {
      for (uint32_t i = 0; i < count; ++i) {
        currentValue_ += *nextDelta++;
        *output++ = currentValue_;
      }
    } else {
      uint16_t restatementBits = chunk;
      for (uint32_t i = 0; i < count; ++i) {
        if (FOLLY_LIKELY(!(restatementBits & 1))) {
          currentValue_ += *nextDelta++;
        } else {
          currentValue_ = *nextRestatement++;
        }
        *output++ = currentValue_;
        restatementBits >>= 1;
      }
    }
    remaining -= count;
  }
}

template <typename T>
template <typename DecoderVisitor>
void DeltaEncoding<T>::readWithVisitor(
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        bool isRestatement;
        isRestatements_->materialize(1, &isRestatement);
        if (isRestatement) {
          restatements_->materialize(1, &currentValue_);
        } else {
          physicalType delta;
          deltas_->materialize(1, &delta);
          currentValue_ += delta;
        }
        return currentValue_;
      });
}

namespace internal {

template <typename physicalType>
void computeDeltas(
    std::span<const physicalType> values,
    Vector<physicalType>* deltas,
    Vector<physicalType>* restatements,
    Vector<bool>* isRestatements) {
  isRestatements->emplace_back(true);
  restatements->emplace_back(values[0]);
  if constexpr (isSignedIntegralType<physicalType>()) {
    for (uint32_t i = 1; i < values.size(); ++i) {
      const bool crossesZero = values[i] > 0 && values[i - 1] < 0;
      if (FOLLY_LIKELY(values[i] >= values[i - 1] && !crossesZero)) {
        isRestatements->emplace_back(false);
        deltas->emplace_back(values[i] - values[i - 1]);
      } else {
        isRestatements->emplace_back(true);
        restatements->emplace_back(values[i]);
      }
    }
  } else {
    for (uint32_t i = 1; i < values.size(); ++i) {
      if (FOLLY_LIKELY(values[i] >= values[i - 1])) {
        isRestatements->emplace_back(false);
        deltas->emplace_back(values[i] - values[i - 1]);
      } else {
        isRestatements->emplace_back(true);
        restatements->emplace_back(values[i]);
      }
    }
  }
}

} // namespace internal

template <typename T>
std::string_view DeltaEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;

  // Fail on empty input.
  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("DeltaEncoding can't be used with 0 rows.");
  }

  const uint32_t rowCount = static_cast<uint32_t>(values.size());
  Vector<physicalType> deltas(&buffer.getMemoryPool());
  Vector<physicalType> restatements(&buffer.getMemoryPool());
  Vector<bool> isRestatements(&buffer.getMemoryPool());

  internal::computeDeltas(values, &deltas, &restatements, &isRestatements);

  ScopedEncodingBuffer tempBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};

  const std::string_view serializedDeltas =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::Delta::Deltas,
          deltas,
          tempBuffer.get(),
          options);
  const std::string_view serializedRestatements =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::Delta::Restatements,
          restatements,
          tempBuffer.get(),
          options);
  const std::string_view serializedIsRestatements =
      selection.template encodeNested<bool>(
          EncodingIdentifiers::Delta::IsRestatements,
          isRestatements,
          tempBuffer.get(),
          options);

  const uint32_t encodingSize =
      Encoding::serializePrefixSize(rowCount, useVarint) + 8 +
      static_cast<uint32_t>(serializedDeltas.size()) +
      static_cast<uint32_t>(serializedRestatements.size()) +
      static_cast<uint32_t>(serializedIsRestatements.size());

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Delta, TypeTraits<T>::dataType, rowCount, useVarint, pos);

  // Data layout (after prefix):
  // 4 bytes: restatement relative offset (X = serializedDeltas.size())
  // 4 bytes: is-restatement relative offset (Y = serializedRestatements.size())
  // X bytes: delta encoding bytes
  // Y bytes: restatement encoding bytes
  // Z bytes: is-restatement encoding bytes
  encoding::writeUint32(static_cast<uint32_t>(serializedDeltas.size()), pos);
  encoding::writeUint32(
      static_cast<uint32_t>(serializedRestatements.size()), pos);
  encoding::writeBytes(serializedDeltas, pos);
  encoding::writeBytes(serializedRestatements, pos);
  encoding::writeBytes(serializedIsRestatements, pos);
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string DeltaEncoding<T>::debugString(int offset) const {
  std::string log = Encoding::debugString(offset);
  log += fmt::format(
      "\n{}deltas child:\n{}",
      std::string(offset + 2, ' '),
      deltas_->debugString(offset + 4));
  log += fmt::format(
      "\n{}restatements child:\n{}",
      std::string(offset + 2, ' '),
      restatements_->debugString(offset + 4));
  log += fmt::format(
      "\n{}isRestatements child:\n{}",
      std::string(offset + 2, ' '),
      isRestatements_->debugString(offset + 4));
  return log;
}

} // namespace facebook::nimble
