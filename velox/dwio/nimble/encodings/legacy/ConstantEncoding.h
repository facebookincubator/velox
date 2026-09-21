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
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/legacy/Encoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Encodes data that is constant, i.e. there is a single unique value.

namespace facebook::nimble::legacy {

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// X bytes: the constant value via encoding primitive.
template <typename T>
class ConstantEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  ConstantEncoding(
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  void materializeBoolsAsBits(uint32_t rowCount, uint64_t* buffer, int begin)
      final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer);

  std::string debugString(int offset) const final;

 private:
  physicalType value_;
};

//
// End of public API. Implementation follows.
//

template <typename T>
ConstantEncoding<T>::ConstantEncoding(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> /* stringBufferFactory */)
    : TypedEncoding<T, physicalType>(memoryPool, data) {
  const char* pos = data.data() + EncodingPrefix::kFixedPrefixSize;
  value_ = encoding::read<physicalType>(pos);
  NIMBLE_CHECK(
      pos == data.data() + data.size(), "Unexpected constant encoding end");
}

template <typename T>
void ConstantEncoding<T>::reset() {}

template <typename T>
void ConstantEncoding<T>::skip(uint32_t /* rowCount */) {}

template <typename T>
void ConstantEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  physicalType* castBuffer = static_cast<physicalType*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    castBuffer[i] = value_;
  }
}

template <>
inline void ConstantEncoding<bool>::materializeBoolsAsBits(
    uint32_t rowCount,
    uint64_t* buffer,
    int begin) {
  velox::bits::fillBits(buffer, begin, begin + rowCount, value_);
}

template <typename T>
void ConstantEncoding<T>::materializeBoolsAsBits(
    uint32_t /*rowCount*/,
    uint64_t* /*buffer*/,
    int /*begin*/) {
  NIMBLE_UNREACHABLE("");
}

template <typename T>
template <typename V>
void ConstantEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(visitor, params, nullptr, [&] { return value_; });
}

template <typename T>
std::string_view ConstantEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer) {
  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("ConstantEncoding cannot be empty.");
  }

  if (selection.statistics().uniqueCounts().value().size() != 1) {
    NIMBLE_INCOMPATIBLE_ENCODING("ConstantEncoding requires constant data.");
  }

  const uint32_t rowCount = values.size();
  uint32_t encodingSize = EncodingPrefix::kFixedPrefixSize;
  if constexpr (isStringType<physicalType>()) {
    encodingSize += 4 + values[0].size();
  } else {
    encodingSize += sizeof(physicalType);
  }
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Constant, TypeTraits<T>::dataType, rowCount, false, pos);
  encoding::write<physicalType>(values[0], pos);
  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string ConstantEncoding<T>::debugString(int offset) const {
  return fmt::format(
      "{}{}<{}> rowCount={} value={}",
      std::string(offset, ' '),
      toString(Encoding::encodingType()),
      toString(Encoding::dataType()),
      Encoding::rowCount(),
      NIMBLE_AS_CONST(T, value_));
}

} // namespace facebook::nimble::legacy
