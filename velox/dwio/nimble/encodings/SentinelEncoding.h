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
#include "folly/container/F14Set.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

#include <optional>

// A sentinel encoding chooses a value not present in the non-null values to
// represent a null value, and then encodes the data (with the sentinel value
// inserted into each slot that was null) as a normal encoding. The 'nullness'
// of the sentinel is handled in the read operations.

namespace facebook::nimble {

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 4 bytes: num nulls
// 4 bytes: non-null child encoding size (X)
// X bytes: non-null child encoding bytes
// Y bytes: type-dependent sentinel value
template <typename T>
class SentinelEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  SentinelEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  uint32_t nullCount() const final;
  bool isNullable() const final;

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;
  uint32_t materializeNullable(
      uint32_t rowCount,
      void* buffer,
      std::function<void*()> getOutputNulls,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr,
      uint32_t offset = 0) final;

  // // Our signature here for estimate size and serialize are a little
  // different
  // // than for non-nullable encodings, as we take in both the nulls and non
  // // nulls. Remember that the size of the values must be equal to the number
  // of
  // // true values in |nulls|.
  // //
  // // Note that nulls[i] is set to true if the ith value is NOT null.
  // static bool estimateSize(
  //     velox::memory::MemoryPool& pool,
  //     std::span<const T> nonNullValues,
  //     std::span<const bool> nulls,
  //     OptimalSearchParams optimalSearchParams,
  //     encodings::EncodingParameters& encodingParameters,
  //     uint32_t* size);

  // static std::string_view serialize(
  //     std::span<const T> nonNullValues,
  //     std::span<const bool> nulls,
  //     const encodings::EncodingParameters& encodingParameters,
  //     Buffer* buffer);

  // // Estimates the best sentinel encoding using default search parameters and
  // // then serializes.
  // static std::string_view serialize(
  //     std::span<const T> nonNullValues,
  //     std::span<const bool> nulls,
  //     Buffer* buffer);

  std::string debugString(int offset) const final;

 private:
  std::unique_ptr<Encoding> sentineledData_;
  physicalType sentinelValue_;
  uint32_t nullCount_;
};

template <typename T>
SentinelEncoding<T>::SentinelEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> /* stringBufferFactory */)
    : TypedEncoding<T, physicalType>(pool, data) {
  const char* pos = data.data() + EncodingPrefix::kFixedPrefixSize;
  nullCount_ = encoding::readUint32(pos);
  const uint32_t sentineledBytes = encoding::readUint32(pos);
  sentineledData_ = deserializeEncoding(this->pool_, {pos, sentineledBytes});
  pos += sentineledBytes;
  sentinelValue_ = encoding::read<physicalType>(pos);
  NIMBLE_CHECK(pos == data.end(), "Unexpected sentinel encoding end");
}

template <typename T>
uint32_t SentinelEncoding<T>::nullCount() const {
  return nullCount_;
}

template <typename T>
bool SentinelEncoding<T>::isNullable() const {
  return true;
}

template <typename T>
void SentinelEncoding<T>::reset() {
  sentineledData_->reset();
}

template <typename T>
void SentinelEncoding<T>::skip(uint32_t rowCount) {
  sentineledData_->skip(rowCount);
}

template <typename T>
void SentinelEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  sentineledData_->materialize(rowCount, buffer);
  if (sentinelValue_ == physicalType()) {
    return;
  }
  physicalType* castBuffer = static_cast<physicalType*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    if (castBuffer[i] == sentinelValue_) {
      castBuffer[i] = physicalType();
    }
  }
}

template <typename T>
uint32_t SentinelEncoding<T>::materializeNullable(
    uint32_t rowCount,
    void* buffer,
    std::function<void*()> getOutputNulls,
    const velox::bits::Bitmap* scatterOutputBitmap,
    uint32_t offset) {
  if (offset > 0) {
    buffer = static_cast<physicalType*>(buffer) + offset;
  }
  sentineledData_->materialize(rowCount, buffer);

  // Member variables in tight loops are bad.
  const physicalType localSentinel = sentinelValue_;
  auto scatterCount =
      scatterOutputBitmap ? scatterOutputBitmap->size() - offset : rowCount;
  void* nullBitmap = getOutputNulls();
  velox::bits::BitmapBuilder nullBits{nullBitmap, offset + scatterCount};
  nullBits.clear(offset, offset + scatterCount);

  uint32_t nonNullCount = 0;
  if (scatterCount != rowCount) {
    physicalType* lastValue = static_cast<physicalType*>(buffer) + rowCount;
    physicalType* castBuffer = static_cast<physicalType*>(buffer);

    for (int64_t i = offset + scatterCount - 1; i >= offset; --i) {
      if (scatterOutputBitmap->test(i)) {
        --lastValue;
        if (*lastValue != localSentinel) {
          castBuffer[i] = *lastValue;
          nullBits.set(i);
          ++nonNullCount;
        }
      }
    }
  } else {
    physicalType* castBuffer = static_cast<physicalType*>(buffer);
    for (uint32_t i = 0; i < rowCount; ++i) {
      auto notNull = *castBuffer++ != localSentinel;
      nonNullCount += static_cast<uint32_t>(notNull);
      nullBits.maybeSet(i, notNull);
    }
  }

  return nonNullCount;
}

namespace {
template <size_t BytesInNumber>
inline uint64_t maxUniqueCount() {
  return 1L << (8 * BytesInNumber);
}

template <>
inline uint64_t maxUniqueCount<8L>() {
  // can't do 1 << 64 because it will generate run time failure of:
  // runtime error: shift exponent 64 is too large for 64-bit type 'long'
  return std::numeric_limits<uint64_t>::max();
}
} // namespace

template <typename physicalType>
std::optional<physicalType> findSentinelValue(
    std::span<const physicalType> nonNullValues,
    std::string* /*unused*/) {
  static_assert(isNumericType<physicalType>());
  folly::F14FastSet<physicalType> uniques(
      nonNullValues.begin(), nonNullValues.end());
  if (UNLIKELY(uniques.size() >= maxUniqueCount<sizeof(physicalType)>())) {
    return std::nullopt;
  }
  physicalType sentinel = physicalType();
  while (uniques.find(sentinel) != uniques.end()) {
    ++sentinel;
  }
  return sentinel;
}

template <>
inline std::optional<std::string_view> findSentinelValue<std::string_view>(
    std::span<const std::string_view> nonNullValues,
    std::string* sentinel) {
  auto intToStringSentinel = [](int i) {
    // Mildly inefficient, but really unlikely to ever matter.
    std::string result;
    while (i) {
      result += char(i % 256);
      i >>= 8;
    }
    return result;
  };
  int next = 1;
loop_start:
  for (const auto value : nonNullValues) {
    if (value == *sentinel) {
      *sentinel = intToStringSentinel(next++);
      goto loop_start;
    }
  }
  return *sentinel;
}

template <>
inline std::optional<std::string> findSentinelValue<std::string>(
    std::span<const std::string> nonNullValues,
    std::string* sentinel) {
  auto intToStringSentinel = [](int i) {
    // Mildly inefficient, but really unlikely to ever matter.
    std::string result;
    while (i) {
      result += char(i % 256);
      i >>= 8;
    }
    return result;
  };
  int next = 1;
loop_start:
  for (const auto& value : nonNullValues) {
    if (value == *sentinel) {
      *sentinel = intToStringSentinel(next++);
      goto loop_start;
    }
  }
  return *sentinel;
}

template <typename physicalType>
Vector<physicalType> createSentineledData(
    velox::memory::MemoryPool& pool,
    std::span<const physicalType> nonNullValues,
    std::span<const bool> nulls,
    physicalType sentinelValue) {
  Vector<physicalType> sentineledData(&pool, nulls.size());
  auto it = nonNullValues.begin();
  for (int i = 0; i < nulls.size(); ++i) {
    if (nulls[i]) {
      sentineledData[i] = *it++;
    } else {
      sentineledData[i] = sentinelValue;
    }
  }
  return sentineledData;
}

// template <typename T>
// std::string_view SentinelEncoding<T>::serialize(
//     std::span<const T> nonNullDataValues,
//     std::span<const bool> nulls,
//     const encodings::EncodingParameters& encodingParameters,
//     Buffer* buffer) {
//   NIMBLE_CHECK(
//       encodingParameters.getType() ==
//               encodings::EncodingParameters::Type::sentinel &&
//           encodingParameters.sentinel_ref().has_value() &&
//           encodingParameters.sentinel_ref()->valuesParameters().has_value(),
//       "Incomplete or incompatible Sentinel encoding parameters.");

//   std::string sentinelHolder; // Used only for string-view type.
//   auto& sentinelParameters = encodingParameters.sentinel_ref().value();
//   // TODO: Once we have a way to store cached values next to encodings, we
//   // should store the previously calculated sentinel value and try to use it
//   in
//   // following serializations.
//   auto nonNullValues =
//       EncodingPhysicalType<T>::asEncodingPhysicalTypeSpan(nonNullDataValues);
//   auto sentinelOptional = findSentinelValue(nonNullValues, &sentinelHolder);
//   if (!sentinelOptional.has_value()) {
//     NIMBLE_INCOMPATIBLE_ENCODING(
//         "Cannot use SentinelEncoding when no value is left for sentinel.");
//   }
//   auto sentinelValue = sentinelOptional.value();
//   auto& pool = buffer->getMemoryPool();
//   const Vector<physicalType> sentineledData =
//       createSentineledData(pool, nonNullValues, nulls, sentinelValue);
//   const uint32_t nullCount =
//       nulls.size() - std::accumulate(nulls.begin(), nulls.end(), 0u);
//   std::string_view sentineledEncoding = serializeEncoding<physicalType>(
//       sentineledData, sentinelParameters.valuesParameters().value(), buffer);
//   uint32_t encodingSize = EncodingPrefix::kFixedPrefixSize + 8 +
//   sentineledEncoding.size(); if constexpr (isNumericType<physicalType>()) {
//     encodingSize += sizeof(physicalType);
//   } else {
//     encodingSize += 4 + sentinelValue.size();
//   }
//   char* reserved = buffer->reserve(encodingSize);
//   char* pos = reserved;
//   Encoding::serializePrefix(
//       EncodingType::Sentinel,
//       TypeTraits<T>::dataType,
//       sentineledData.size(),
//       pos);
//   encoding::writeUint32(nullCount, pos);
//   encoding::writeString(sentineledEncoding, pos);
//   encoding::write<physicalType>(sentinelValue, pos);
//   NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
//   return {reserved, encodingSize};
// }

// template <typename T>
// bool SentinelEncoding<T>::estimateSize(
//     velox::memory::MemoryPool& pool,
//     std::span<const T> nonNullDataValues,
//     std::span<const bool> nulls,
//     OptimalSearchParams optimalSearchParams,
//     encodings::EncodingParameters& encodingParameters,
//     uint32_t* size) {
//   std::string holder;
//   auto nonNullValues =
//       EncodingPhysicalType<T>::asEncodingPhysicalTypeSpan(nonNullDataValues);
//   auto sentinelOptional = findSentinelValue(nonNullValues, &holder);
//   if (!sentinelOptional.has_value()) {
//     NIMBLE_INCOMPATIBLE_ENCODING(
//         "Cannot use SentinelEncoding when no value is left for sentinel.");
//   }
//   auto sentinelValue = sentinelOptional.value();
//   const Vector<physicalType> sentineledData =
//       createSentineledData(pool, nonNullValues, nulls, sentinelValue);
//   uint32_t sentineledSize;
//   auto& sentinelParameters = encodingParameters.set_sentinel();
//   estimateOptimalEncodingSize<physicalType>(
//       pool,
//       sentineledData,
//       optimalSearchParams,
//       &sentineledSize,
//       sentinelParameters.valuesParameters().ensure());
//   *size = EncodingPrefix::kFixedPrefixSize + 8 + sentineledSize +
//   sizeof(physicalType); return true;
// }

// template <typename T>
// std::string_view SentinelEncoding<T>::serialize(
//     std::span<const T> nonNullValues,
//     std::span<const bool> nulls,
//     Buffer* buffer) {
//   encodings::EncodingParameters encodingParameters;
//   uint32_t unusedSize;
//   SentinelEncoding<T>::estimateSize(
//       buffer->getMemoryPool(),
//       nonNullValues,
//       nulls,
//       OptimalSearchParams(),
//       encodingParameters,
//       &unusedSize);
//   return SentinelEncoding<T>::serialize(
//       nonNullValues, nulls, encodingParameters, buffer);
// }

template <typename T>
std::string SentinelEncoding<T>::debugString(int offset) const {
  std::string log = Encoding::debugString(offset);
  log += fmt::format(
      "\n{}sentineled child:\n{}",
      std::string(offset + 2, ' '),
      sentineledData_->debugString(offset + 4));
  log += fmt::format(
      "\n{}num nulls: {}", std::string(offset + 2, ' '), nullCount_);
  log += fmt::format(
      "\n{}sentinel value: {}", std::string(offset + 2, ' '), sentinelValue_);
  return log;
}

} // namespace facebook::nimble
