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

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Stores integer data in a delta encoding. We use three child encodings:
// one for whether each row is a delta from the last or a restatement,
// one for the deltas, and one one for the restatements. For now we
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

namespace facebook::nimble::legacy {

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
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  std::string debugString(int offset) const final;

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer);

 private:
  physicalType currentValue_;
  std::unique_ptr<Encoding> deltas_;
  std::unique_ptr<Encoding> restatements_;
  std::unique_ptr<Encoding> isRestatements_;
  // Temporary bufs.
  Vector<physicalType> deltasBuffer_;
  Vector<physicalType> restatementsBuffer_;
  Vector<bool> isRestatementsBuffer_;
};

//
// End of public API. Implementation follows.
//

template <typename T>
DeltaEncoding<T>::DeltaEncoding(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory)
    : TypedEncoding<T, physicalType>(memoryPool, data),
      deltasBuffer_(&memoryPool),
      restatementsBuffer_(&memoryPool),
      isRestatementsBuffer_(&memoryPool) {
  const EncodingFactory factory;
  auto pos = data.data() + EncodingPrefix::kFixedPrefixSize;
  const uint32_t restatementsOffset = encoding::readUint32(pos);
  const uint32_t isRestatementsOffset = encoding::readUint32(pos);
  deltas_ = factory.create(
      memoryPool,
      {pos, restatementsOffset},
      stringBufferFactory,
      Encoding::Options{});
  pos += restatementsOffset;
  restatements_ = factory.create(
      memoryPool,
      {pos, isRestatementsOffset},
      stringBufferFactory,
      Encoding::Options{});
  pos += isRestatementsOffset;
  isRestatements_ = factory.create(
      memoryPool,
      {pos, static_cast<size_t>(data.end() - pos)},
      stringBufferFactory,
      Encoding::Options{});
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
  isRestatementsBuffer_.resize(rowCount);
  isRestatements_->materialize(rowCount, isRestatementsBuffer_.data());
  const uint32_t numRestatements = std::accumulate(
      isRestatementsBuffer_.begin(), isRestatementsBuffer_.end(), 0UL);
  // Find the last restatement, then accumulate deltas forward from it.
  int64_t lastRestatement = rowCount - 1;
  while (lastRestatement >= 0) {
    if (isRestatementsBuffer_[lastRestatement]) {
      break;
    }
    --lastRestatement;
  }
  if (lastRestatement >= 0) {
    restatements_->skip(numRestatements - 1);
    restatements_->materialize(1, &currentValue_);
    const uint32_t deltasToSkip = lastRestatement -
        std::accumulate(isRestatementsBuffer_.begin(),
                        isRestatementsBuffer_.begin() + lastRestatement,
                        0UL);
    deltas_->skip(deltasToSkip);
  }
  const uint32_t deltasToAccumulate = rowCount - 1 - lastRestatement;
  deltasBuffer_.resize(deltasToAccumulate);
  deltas_->materialize(deltasToAccumulate, deltasBuffer_.data());
  currentValue_ += std::accumulate(
      deltasBuffer_.begin(), deltasBuffer_.end(), physicalType());
}

template <typename T>
void DeltaEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  isRestatementsBuffer_.resize(rowCount);
  isRestatements_->materialize(rowCount, isRestatementsBuffer_.data());
  const uint32_t numRestatements = std::accumulate(
      isRestatementsBuffer_.begin(), isRestatementsBuffer_.end(), 0UL);
  restatementsBuffer_.reserve(numRestatements);
  restatements_->materialize(numRestatements, restatementsBuffer_.data());
  deltasBuffer_.reserve(rowCount - numRestatements);
  deltas_->materialize(rowCount - numRestatements, deltasBuffer_.data());
  physicalType* castValue = static_cast<physicalType*>(buffer);
  physicalType* nextRestatement = restatementsBuffer_.begin();
  physicalType* nextDelta = deltasBuffer_.begin();
  for (uint32_t i = 0; i < rowCount; ++i) {
    if (isRestatementsBuffer_[i]) {
      currentValue_ = *nextRestatement++;
    } else {
      currentValue_ += *nextDelta++;
    }
    *castValue++ = currentValue_;
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
    Buffer& buffer) {
  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("DeltaEncoding can't be used with 0 rows.");
  }

  const uint32_t rowCount = static_cast<uint32_t>(values.size());
  Vector<physicalType> deltas(&buffer.getMemoryPool());
  Vector<physicalType> restatements(&buffer.getMemoryPool());
  Vector<bool> isRestatements(&buffer.getMemoryPool());

  internal::computeDeltas(values, &deltas, &restatements, &isRestatements);

  Buffer tempBuffer{buffer.getMemoryPool()};

  const std::string_view serializedDeltas =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::Delta::Deltas, deltas, tempBuffer);
  const std::string_view serializedRestatements =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::Delta::Restatements, restatements, tempBuffer);
  const std::string_view serializedIsRestatements =
      selection.template encodeNested<bool>(
          EncodingIdentifiers::Delta::IsRestatements,
          isRestatements,
          tempBuffer);

  const uint32_t encodingSize = EncodingPrefix::kFixedPrefixSize + 8 +
      static_cast<uint32_t>(serializedDeltas.size()) +
      static_cast<uint32_t>(serializedRestatements.size()) +
      static_cast<uint32_t>(serializedIsRestatements.size());
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Delta, TypeTraits<T>::dataType, rowCount, false, pos);
  encoding::writeUint32(static_cast<uint32_t>(serializedDeltas.size()), pos);
  encoding::writeUint32(
      static_cast<uint32_t>(serializedRestatements.size()), pos);
  encoding::writeBytes(serializedDeltas, pos);
  encoding::writeBytes(serializedRestatements, pos);
  encoding::writeBytes(serializedIsRestatements, pos);
  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
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

} // namespace facebook::nimble::legacy
