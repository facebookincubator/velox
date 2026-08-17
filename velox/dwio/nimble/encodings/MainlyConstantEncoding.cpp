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

#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"

namespace facebook::nimble {

MainlyConstantEncoding<std::string_view>::MainlyConstantEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : MainlyConstantEncodingBase<std::string_view>(pool, data, options) {
  const EncodingFactory factory{options};
  const char* pos = data.data() + this->dataOffset();
  const uint32_t isCommonBytes = encoding::readUint32(pos);
  isCommon_ =
      factory.create(*this->pool_, {pos, isCommonBytes}, stringBufferFactory);
  pos += isCommonBytes;
  const uint32_t otherValuesBytes = encoding::readUint32(pos);
  otherValues_ = factory.create(
      *this->pool_, {pos, otherValuesBytes}, stringBufferFactory);
  pos += otherValuesBytes;
  commonValue_ = encoding::read<physicalType>(pos);
  NIMBLE_CHECK(pos == data.end(), "Unexpected mainly constant encoding end");
  auto stringBuffer =
      static_cast<char*>(stringBufferFactory(commonValue_.size()));
  std::memcpy(stringBuffer, commonValue_.data(), commonValue_.size());
  commonValue_ = std::string_view(stringBuffer, commonValue_.size());
}

std::string_view MainlyConstantEncoding<std::string_view>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("MainlyConstantEncoding cannot be empty.");
  }

  const auto& uniqueCounts = selection.statistics().uniqueCounts().value();
  const auto commonElement =
      MainlyConstantEncodingBase<std::string_view>::mainlyConstantCommonValue(
          uniqueCounts);

  const uint32_t entryCount = values.size();

  auto* pool = &buffer.getMemoryPool();
  physicalType commonValue = commonElement->first;
  auto childStreams =
      MainlyConstantEncodingBase<std::string_view>::prepareChildStreams(
          pool, values, commonValue, commonElement->second);

  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  std::string_view serializedIsCommon = selection.template encodeNested<bool>(
      EncodingIdentifiers::MainlyConstant::IsCommon,
      childStreams.isCommon,
      scopedBuffer.get(),
      options);
  std::string_view serializedOtherValues =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::MainlyConstant::OtherValues,
          childStreams.otherValues,
          scopedBuffer.get(),
          options);

  uint32_t encodingSize = Encoding::serializePrefixSize(entryCount, useVarint) +
      8 + serializedIsCommon.size() + serializedOtherValues.size();
  if constexpr (isNumericType<physicalType>()) {
    encodingSize += sizeof(physicalType);
  } else {
    encodingSize += 4 + commonValue.size();
  }
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::MainlyConstant,
      TypeTraits<std::string_view>::dataType,
      entryCount,
      useVarint,
      pos);
  encoding::writeString(serializedIsCommon, pos);
  encoding::writeString(serializedOtherValues, pos);
  encoding::write<physicalType>(commonValue, pos);
  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}
} // namespace facebook::nimble
