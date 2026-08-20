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
#include "velox/dwio/nimble/tools/EncodingUtilities.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryTypes.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"

namespace facebook::nimble::tools {
namespace {
uint32_t prefixSize(std::string_view stream, bool useVarintRowCount = false) {
  NIMBLE_CHECK_GE(
      stream.size(),
      EncodingPrefix::kRowCountOffset + 1,
      "Unexpected end of stream.");
  return EncodingPrefix::prefixSize(stream, useVarintRowCount);
}

void extractCompressionType(
    EncodingType encodingType,
    DataType /* dataType */,
    std::string_view stream,
    bool useVarintRowCount,
    std::unordered_map<EncodingPropertyType, EncodingProperty>& properties) {
  switch (encodingType) {
    // Compression type is the byte right after the encoding header for both
    // encodings.
    case EncodingType::Trivial:
    case EncodingType::FixedBitWidth:
    case EncodingType::BlockBitPacking: {
      auto pos = stream.data() + prefixSize(stream, useVarintRowCount);
      properties.insert(
          {EncodingPropertyType::Compression,
           EncodingProperty{
               .value = toString(
                   static_cast<CompressionType>(encoding::readChar(pos))),
           }});
      break;
    }
    case EncodingType::RLE:
    case EncodingType::Dictionary:
    case EncodingType::SharedDictionary:
    case EncodingType::Sentinel:
    case EncodingType::Nullable:
    case EncodingType::SparseBool:
    case EncodingType::Varint:
    case EncodingType::Delta:
    case EncodingType::DeltaBlock:
    case EncodingType::Constant:
    case EncodingType::MainlyConstant:
    case EncodingType::Prefix:
    case EncodingType::ALP:
    case EncodingType::Fsst:
    case EncodingType::PFOR:
    case EncodingType::SimdForBitpack:
    // SubIntSplit integration is disabled; it carries no separate compression
    // byte, so treat it like the other encodings handled here.
    case EncodingType::SubIntSplit:
    case EncodingType::FrequencyPartition:
    case EncodingType::FOR:
    case EncodingType::Huffman:
      break;
  }
}

std::unordered_map<EncodingPropertyType, EncodingProperty>
extractEncodingProperties(
    EncodingType encodingType,
    DataType dataType,
    std::string_view stream,
    bool useVarintRowCount) {
  std::unordered_map<EncodingPropertyType, EncodingProperty> properties{};
  extractCompressionType(
      encodingType, dataType, stream, useVarintRowCount, properties);
  properties.insert(
      {EncodingPropertyType::EncodedSize,
       {.value = folly::to<std::string>(stream.size())}});
  return properties;
}

void traverseEncodings(
    std::string_view stream,
    uint32_t level,
    uint32_t index,
    const std::string& nestedEncodingName,
    bool useVarintRowCount,
    std::function<bool(
        EncodingType,
        DataType,
        uint32_t /* level */,
        uint32_t /* index */,
        std::string /* nestedEncodingName */,
        std::unordered_map<
            EncodingPropertyType,
            EncodingProperty> /* properties */)> visitor) {
  NIMBLE_CHECK_GE(
      stream.size(),
      EncodingPrefix::kRowCountOffset + 1,
      "Unexpected end of stream.");

  const EncodingType encodingType = EncodingPrefix::encodingType(stream);
  auto dataType = EncodingPrefix::dataType(stream);
  const auto dataOffset = prefixSize(stream, useVarintRowCount);
  bool continueTraversal = visitor(
      encodingType,
      dataType,
      level,
      index,
      nestedEncodingName,
      extractEncodingProperties(
          encodingType, dataType, stream, useVarintRowCount));

  if (!continueTraversal) {
    return;
  }

  switch (encodingType) {
    case EncodingType::FixedBitWidth:
    case EncodingType::Varint:
    case EncodingType::Constant:
    case EncodingType::Prefix:
    case EncodingType::DeltaBlock:
    case EncodingType::SimdForBitpack:
    // SubIntSplit integration is disabled; treat it as having no nested
    // encoding to traverse.
    case EncodingType::SubIntSplit: {
      // don't have any nested encoding
      break;
    }
    case EncodingType::ALP: {
      const char* pos = stream.data() + dataOffset;
      const auto header = detail::alp::readHeader(pos);
      uint32_t exceptionCount = 0;
      if (header.hasExceptions) {
        exceptionCount = varint::readVarint32(&pos);
      }
      const uint32_t encodedValuesSize = varint::readVarint32(&pos);
      traverseEncodings(
          {pos, encodedValuesSize},
          level + 1,
          0,
          "EncodedValues",
          useVarintRowCount,
          visitor);
      pos += encodedValuesSize;
      if (exceptionCount > 0) {
        const uint32_t exceptionPositionsSize = varint::readVarint32(&pos);
        traverseEncodings(
            {pos, exceptionPositionsSize},
            level + 1,
            1,
            "ExceptionPositions",
            useVarintRowCount,
            visitor);
        pos += exceptionPositionsSize;
        const uint32_t exceptionValuesSize = varint::readVarint32(&pos);
        traverseEncodings(
            {pos, exceptionValuesSize},
            level + 1,
            2,
            "ExceptionValues",
            useVarintRowCount,
            visitor);
      }
      break;
    }
    case EncodingType::BlockBitPacking: {
      // Layout after the common prefix: compressionType [1 byte], blockSize
      // [varint], numBlocks [varint], then three self-describing nested
      // metadata sub-streams, each preceded by a varint size: the per-block
      // baselines, bit widths, and data offsets.
      const char* pos = stream.data() + dataOffset;
      encoding::readChar(pos); // compressionType
      varint::readVarint32(&pos); // blockSize
      varint::readVarint32(&pos); // numBlocks
      const uint32_t baselinesSize = varint::readVarint32(&pos);
      if (baselinesSize > 0) {
        traverseEncodings(
            {pos, baselinesSize},
            level + 1,
            0,
            "Baselines",
            useVarintRowCount,
            visitor);
      }
      pos += baselinesSize;
      const uint32_t bitWidthsSize = varint::readVarint32(&pos);
      if (bitWidthsSize > 0) {
        traverseEncodings(
            {pos, bitWidthsSize},
            level + 1,
            1,
            "BitWidths",
            useVarintRowCount,
            visitor);
      }
      pos += bitWidthsSize;
      const uint32_t offsetsSize = varint::readVarint32(&pos);
      if (offsetsSize > 0) {
        traverseEncodings(
            {pos, offsetsSize},
            level + 1,
            2,
            "DataOffsets",
            useVarintRowCount,
            visitor);
      }
      break;
    }
    case EncodingType::PFOR: {
      // Layout after the common prefix: baseline [dataTypeSize bytes],
      // baseBitWidth [1 byte], numExceptions [varint], then two
      // self-describing nested sub-streams, each preceded by a varint size:
      // the exception positions and the exception residual values.
      const char* pos = stream.data() + dataOffset;
      pos += detail::dataTypeSize(dataType); // baseline
      encoding::readChar(pos); // baseBitWidth
      varint::readVarint32(&pos); // numExceptions
      const uint32_t positionsSize = varint::readVarint32(&pos);
      if (positionsSize > 0) {
        traverseEncodings(
            {pos, positionsSize},
            level + 1,
            0,
            "ExceptionPositions",
            useVarintRowCount,
            visitor);
      }
      pos += positionsSize;
      const uint32_t valuesSize = varint::readVarint32(&pos);
      if (valuesSize > 0) {
        traverseEncodings(
            {pos, valuesSize},
            level + 1,
            1,
            "ExceptionValues",
            useVarintRowCount,
            visitor);
      }
      break;
    }
    case EncodingType::Trivial: {
      if (dataType == DataType::String) {
        const char* pos = stream.data() + dataOffset + 1;
        const uint32_t lengthsBytes = encoding::readUint32(pos);
        traverseEncodings(
            {pos, lengthsBytes},
            level + 1,
            0,
            "Lengths",
            useVarintRowCount,
            visitor);
      }
      break;
    }
    case EncodingType::Fsst: {
      traverseEncodings(
          FsstEncoding::lengthsEncoding(stream),
          level + 1,
          0,
          "Lengths",
          useVarintRowCount,
          visitor);
      break;
    }
    case EncodingType::SparseBool: {
      const char* pos = stream.data() + dataOffset + 1;
      traverseEncodings(
          {pos, stream.size() - (pos - stream.data())},
          level + 1,
          0,
          "Indices",
          useVarintRowCount,
          visitor);
      break;
    }
    case EncodingType::MainlyConstant: {
      const char* pos = stream.data() + dataOffset;
      const uint32_t isCommonBytes = encoding::readUint32(pos);
      traverseEncodings(
          {pos, isCommonBytes},
          level + 1,
          0,
          "IsCommon",
          useVarintRowCount,
          visitor);
      pos += isCommonBytes;
      const uint32_t otherValueBytes = encoding::readUint32(pos);
      traverseEncodings(
          {pos, otherValueBytes},
          level + 1,
          1,
          "OtherValues",
          useVarintRowCount,
          visitor);
      break;
    }
    case EncodingType::Dictionary: {
      const char* pos = stream.data() + dataOffset;
      const uint32_t alphabetBytes = encoding::readUint32(pos);
      traverseEncodings(
          {pos, alphabetBytes},
          level + 1,
          0,
          "Alphabet",
          useVarintRowCount,
          visitor);
      pos += alphabetBytes;
      traverseEncodings(
          {pos, stream.size() - (pos - stream.data())},
          level + 1,
          1,
          "Indices",
          useVarintRowCount,
          visitor);
      break;
    }
    case EncodingType::SharedDictionary: {
      const char* pos = stream.data() + prefixSize(stream, useVarintRowCount);
      readSharedDictionaryScope(stream, pos);
      readSharedDictionaryId(stream, pos);
      const auto indicesOffset = static_cast<size_t>(pos - stream.data());
      NIMBLE_CHECK_LT(
          indicesOffset,
          stream.size(),
          "Shared dictionary encoding is missing its indices.");
      traverseEncodings(
          {pos, stream.size() - indicesOffset},
          level + 1,
          0,
          "Indices",
          useVarintRowCount,
          visitor);
      break;
    }
    case EncodingType::RLE: {
      const char* pos = stream.data() + dataOffset;
      const uint32_t runLengthBytes = encoding::readUint32(pos);
      traverseEncodings(
          {pos, runLengthBytes},
          level + 1,
          0,
          "Lengths",
          useVarintRowCount,
          visitor);
      if (dataType != DataType::Bool) {
        pos += runLengthBytes;
        traverseEncodings(
            {pos, stream.size() - (pos - stream.data())},
            level + 1,
            1,
            "Values",
            useVarintRowCount,
            visitor);
      }
      break;
    }
    case EncodingType::Delta: {
      const char* pos = stream.data() + dataOffset;
      const uint32_t deltaBytes = encoding::readUint32(pos);
      const uint32_t restatementBytes = encoding::readUint32(pos);
      traverseEncodings(
          {pos, deltaBytes},
          level + 1,
          0,
          "Deltas",
          useVarintRowCount,
          visitor);
      pos += deltaBytes;
      traverseEncodings(
          {pos, restatementBytes},
          level + 1,
          1,
          "Restatements",
          useVarintRowCount,
          visitor);
      pos += restatementBytes;
      traverseEncodings(
          {pos, stream.size() - (pos - stream.data())},
          level + 1,
          2,
          "IsRestatements",
          useVarintRowCount,
          visitor);
      break;
    }
    case EncodingType::Nullable: {
      const char* pos = stream.data() + dataOffset;
      const uint32_t dataBytes = encoding::readUint32(pos);
      traverseEncodings(
          {pos, dataBytes}, level + 1, 0, "Data", useVarintRowCount, visitor);
      pos += dataBytes;
      traverseEncodings(
          {pos, stream.size() - (pos - stream.data())},
          level + 1,
          1,
          "Nulls",
          useVarintRowCount,
          visitor);
      break;
    }
    case EncodingType::Sentinel: {
      const char* pos = stream.data() + dataOffset + 8;
      traverseEncodings(
          {pos, stream.size() - (pos - stream.data())},
          level + 1,
          0,
          "Sentinels",
          useVarintRowCount,
          visitor);
      break;
    }
    case EncodingType::FrequencyPartition:
    case EncodingType::FOR:
    case EncodingType::Huffman:
      break;
  }
}

} // namespace

std::ostream& operator<<(std::ostream& out, EncodingPropertyType propertyType) {
  switch (propertyType) {
    case EncodingPropertyType::Compression:
      return out << "Compression";
    default:
      return out << "Unknown";
  }
}

void traverseEncodings(
    std::string_view stream,
    std::function<bool(
        EncodingType,
        DataType,
        uint32_t /* level */,
        uint32_t /* index */,
        std::string /* nestedEncodingName */,
        std::unordered_map<
            EncodingPropertyType,
            EncodingProperty> /* properties */)> visitor) {
  traverseEncodings(stream, 0, 0, "", /*useVarintRowCount=*/false, visitor);
}

std::string getStreamInputLabel(nimble::ChunkedStream& stream) {
  std::string label;
  uint32_t chunkId = 0;
  while (stream.hasNext()) {
    label += folly::to<std::string>(chunkId++);
    auto compression = stream.peekCompressionType();
    if (compression != CompressionType::Uncompressed) {
      label += "{" + toString(compression) + "}";
    }
    label += ":" + getEncodingLabel(stream.nextChunk());

    if (stream.hasNext()) {
      label += ";";
    }
  }
  return label;
}

std::string getEncodingLabel(std::string_view stream) {
  std::string label;
  uint32_t currentLevel = 0;

  traverseEncodings(
      stream,
      [&](EncodingType encodingType,
          DataType dataType,
          uint32_t level,
          uint32_t index,
          std::string nestedEncodingName,
          std::unordered_map<EncodingPropertyType, EncodingProperty> properties)
          -> bool {
        if (level > currentLevel) {
          label += "[" + nestedEncodingName + ":";
        } else if (level < currentLevel) {
          label += "]";
        }

        if (index > 0) {
          label += "," + nestedEncodingName + ":";
        }

        currentLevel = level;

        label += toString(encodingType) + "<" + toString(dataType);
        const auto& encodedSize =
            properties.find(EncodingPropertyType::EncodedSize);
        if (encodedSize != properties.end()) {
          label += "," + encodedSize->second.value;
        }
        label += ">";

        const auto& compression =
            properties.find(EncodingPropertyType::Compression);
        if (compression != properties.end()
            // If the "Uncompressed" label clutters the output, uncomment the
            // next line.
            // && compression->second.value !=
            // toString(CompressionType::Uncompressed)
        ) {
          label += "{" + compression->second.value + "}";
        }

        return true;
      });
  while (currentLevel-- > 0) {
    label += "]";
  }

  return label;
}

} // namespace facebook::nimble::tools
