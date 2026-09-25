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
#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"

namespace facebook::nimble {

ALPRDEncodingBase::Metadata ALPRDEncodingBase::readMetadata(
    std::string_view data,
    const Encoding::Options& options) {
  const auto prefix = EncodingPrefix::consume(data, options.useVarintRowCount);
  NIMBLE_CHECK_EQ(EncodingPrefix::encodingType(prefix), EncodingType::ALPRD);
  Metadata metadata{};
  metadata.dataType = EncodingPrefix::dataType(prefix);
  NIMBLE_CHECK(
      metadata.dataType == DataType::Float ||
          metadata.dataType == DataType::Double,
      "ALPRD requires a floating-point type.");
  metadata.rowCount =
      EncodingPrefix::readRowCount(prefix, options.useVarintRowCount);
  NIMBLE_CHECK_GT(metadata.rowCount, 0, "Empty ALPRD encoding.");
  auto& parameters = metadata.parameters;
  parameters.rightBitWidth = encoding::readByte(data);
  parameters.dictionarySize = encoding::readByte(data);
  const auto valueBitWidth = metadata.valueBitWidth();
  NIMBLE_CHECK_GE(parameters.rightBitWidth, valueBitWidth - kMaxHighBitWidth);
  NIMBLE_CHECK_LT(parameters.rightBitWidth, valueBitWidth);
  NIMBLE_CHECK_GT(parameters.dictionarySize, 0);
  NIMBLE_CHECK_LE(parameters.dictionarySize, kMaxDictionarySize);
  metadata.exceptionCount = encoding::readVarint32(data);
  NIMBLE_CHECK_LE(metadata.exceptionCount, metadata.rowCount);

  const auto highLimit = metadata.highLimit();
  for (uint8_t i = 0; i < parameters.dictionarySize; ++i) {
    const auto high = encoding::readUint16(data);
    NIMBLE_CHECK_LT(high, highLimit, "Invalid ALPRD dictionary value.");
    for (uint8_t j = 0; j < i; ++j) {
      NIMBLE_CHECK_NE(high, parameters.dictionary[j], "Duplicate ALPRD key.");
    }
    parameters.dictionary[i] = high;
  }

  const std::array<DataType, 4> childTypes{
      DataType::Uint16,
      metadata.dataType == DataType::Float ? DataType::Uint32
                                           : DataType::Uint64,
      DataType::Uint32,
      DataType::Uint16,
  };
  for (uint8_t i = 0; i < metadata.childrenCount(); ++i) {
    const auto child = encoding::readLengthPrefixedBytes(data);
    auto cursor = child;
    const auto childPrefix =
        EncodingPrefix::consume(cursor, options.useVarintRowCount);
    NIMBLE_CHECK_EQ(
        EncodingPrefix::dataType(childPrefix),
        childTypes[i],
        "Invalid ALPRD child type.");
    NIMBLE_CHECK_EQ(
        EncodingPrefix::readRowCount(childPrefix, options.useVarintRowCount),
        i < 2 ? metadata.rowCount : metadata.exceptionCount,
        "Invalid ALPRD child row count.");
    metadata.children[i] = child;
  }
  NIMBLE_CHECK(data.empty(), "Unexpected bytes after ALPRD children.");
  return metadata;
}

std::unique_ptr<Encoding> ALPRDEncodingBase::createChild(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const Encoding::Options& options) {
  auto child = EncodingFactory(options).create(
      pool, data, [](uint32_t) -> void* { return nullptr; });
  NIMBLE_CHECK(!child->isNullable(), "ALPRD children must be non-nullable.");
  return child;
}

void ALPRDEncodingBase::loadExceptions(
    velox::memory::MemoryPool& pool,
    const Metadata& metadata,
    const Encoding::Options& options,
    Vector<uint32_t>& positions,
    Vector<uint16_t>& highParts) {
  NIMBLE_DCHECK(positions.empty());
  NIMBLE_DCHECK(highParts.empty());
  if (metadata.exceptionCount == 0) {
    return;
  }
  auto positionDecoder = createChild(pool, metadata.children[2], options);
  auto highPartDecoder = createChild(pool, metadata.children[3], options);
  positions.resize(metadata.exceptionCount);
  highParts.resize(metadata.exceptionCount);
  positionDecoder->materialize(metadata.exceptionCount, positions.data());
  highPartDecoder->materialize(metadata.exceptionCount, highParts.data());
  const auto highLimit = metadata.highLimit();
  for (uint32_t i = 0; i < metadata.exceptionCount; ++i) {
    NIMBLE_CHECK_LT(
        positions[i], metadata.rowCount, "Invalid ALPRD exception position.");
    if (i != 0) {
      NIMBLE_CHECK_GT(
          positions[i],
          positions[i - 1],
          "ALPRD exception positions must increase.");
    }
    NIMBLE_CHECK_LT(
        highParts[i], highLimit, "Invalid ALPRD exception high part.");
  }
}

} // namespace facebook::nimble
