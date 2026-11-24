/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/experimental/wave/dwio/nimble/NimbleStream.h"

#include <fmt/format.h>
#include <algorithm>
#include <cstring>
#include <tuple>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"

namespace facebook::velox::wave {

namespace {

using nimble::CompressionType;
using nimble::DataType;
using nimble::EncodingPrefix;
using nimble::EncodingType;

template <typename T>
T readAt(std::string_view data, int64_t offset) {
  VELOX_CHECK_LE(offset + sizeof(T), data.size(), "Truncated Nimble encoding");
  T value;
  memcpy(&value, data.data() + offset, sizeof(T));
  return value;
}

// Returns the width of the physical type of 'dataType' in bits.
int32_t physicalBits(DataType dataType) {
  switch (dataType) { // NOLINT(clang-diagnostic-switch-enum)
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Float:
      return 32;
    case DataType::Int64:
    case DataType::Uint64:
    case DataType::Double:
      return 64;
    default:
      VELOX_NYI(
          "Nimble data type not supported on GPU: {}", toString(dataType));
  }
}

int64_t readPhysical(std::string_view data, int64_t offset, int32_t bits) {
  return bits == 32 ? readAt<uint32_t>(data, offset)
                    : static_cast<int64_t>(readAt<uint64_t>(data, offset));
}

std::string fbwAccessor(int32_t bits) {
  return bits == 32 ? "NFbw<uint32_t>" : "NFbw<uint64_t>";
}

class TreeParser {
 public:
  TreeParser(
      std::string_view stream,
      memory::MemoryPool* pool,
      NimbleStream& result)
      : stream_(stream), pool_(pool), result_(result) {}

  // Parses the value encoding of 'size' bytes at 'offset' in the stream.
  // Returns the node index and the accessor type.
  std::pair<int32_t, std::string> parse(int64_t offset, int64_t size) {
    auto data = encodingAt(offset, size);
    auto encoding = EncodingPrefix::encodingType(data);
    auto dataType = EncodingPrefix::dataType(data);
    NimbleNode node = makeNode(data);
    const int32_t nodeIdx = addNode(node);
    const int64_t pos = EncodingPrefix::kFixedPrefixSize;
    std::string accessor;
    const bool isString = dataType == DataType::String;
    if (isString &&
        (encoding == EncodingType::Trivial ||
         encoding == EncodingType::Constant)) {
      accessor = parseString(data, offset, nodeIdx, node);
      result_.nodes.at(nodeIdx) = node;
      return {nodeIdx, accessor};
    }
    switch (encoding) { // NOLINT(clang-diagnostic-switch-enum)
      case EncodingType::Trivial:
      case EncodingType::FixedBitWidth:
      case EncodingType::Constant:
        setScalar(data, offset, node);
        accessor = encoding == EncodingType::Constant
            ? "NConstant"
            : fbwAccessor(physicalBits(dataType));
        break;
      case EncodingType::Dictionary: {
        checkValueType(dataType);
        auto alphabetSize = readAt<uint32_t>(data, pos);
        auto alphabet = parse(offset + pos + 4, alphabetSize);
        auto indicesOffset = pos + 4 + alphabetSize;
        auto indices = parse(offset + indicesOffset, size - indicesOffset);
        node.children[0] = alphabet.first;
        node.children[1] = indices.first;
        accessor =
            fmt::format("NDict<{}, {}>", indices.second, alphabet.second);
        break;
      }
      case EncodingType::RLE: {
        checkValueType(dataType);
        auto lengthsSize = readAt<uint32_t>(data, pos);
        node.children[0] = parseScalar(offset + pos + 4, lengthsSize);
        auto valuesOffset = pos + 4 + lengthsSize;
        auto values = parse(offset + valuesOffset, size - valuesOffset);
        node.children[1] = values.first;
        accessor = fmt::format("NRle<{}>", values.second);
        result_.prepareNodes.push_back(nodeIdx);
        break;
      }
      case EncodingType::MainlyConstant: {
        auto isCommonSize = readAt<uint32_t>(data, pos);
        node.children[0] = parseBool(offset + pos + 4, isCommonSize);
        auto othersOffset = pos + 4 + isCommonSize;
        auto othersSize = readAt<uint32_t>(data, othersOffset);
        auto others = parse(offset + othersOffset + 4, othersSize);
        node.children[1] = others.first;
        const auto commonOffset = othersOffset + 4 + othersSize;
        if (isString) {
          // The common string is its size and bytes.
          node.value = readAt<uint32_t>(data, commonOffset);
          node.dataOffset = offset + commonOffset + 4;
          accessor = fmt::format("NMainlyConstantString<{}>", others.second);
        } else {
          node.value = readPhysical(data, commonOffset, physicalBits(dataType));
          accessor = fmt::format("NMainlyConstant<{}>", others.second);
        }
        result_.prepareNodes.push_back(nodeIdx);
        break;
      }
      case EncodingType::Nullable: {
        auto valuesSize = readAt<uint32_t>(data, pos);
        auto values = parse(offset + pos + 4, valuesSize);
        node.children[0] = values.first;
        auto nullsOffset = pos + 4 + valuesSize;
        node.children[1] = parseBool(offset + nullsOffset, size - nullsOffset);
        accessor = fmt::format("NNullable<{}>", values.second);
        result_.prepareNodes.push_back(nodeIdx);
        break;
      }
      default:
        VELOX_NYI(
            "Nimble encoding not supported on GPU: {}",
            static_cast<int32_t>(encoding));
    }
    result_.nodes.at(nodeIdx) = node;
    return {nodeIdx, accessor};
  }

  static void checkValueType(DataType dataType) {
    if (dataType != DataType::String) {
      physicalBits(dataType);
    }
  }

  // Sets the fields of a Trivial or Constant string node. A Trivial node gets
  // a prepare op that makes the offsets of the strings from their lengths.
  std::string parseString(
      std::string_view data,
      int64_t offset,
      int32_t nodeIdx,
      NimbleNode& node) {
    const int64_t pos = EncodingPrefix::kFixedPrefixSize;
    if (EncodingPrefix::encodingType(data) == EncodingType::Constant) {
      node.value = readAt<uint32_t>(data, pos);
      node.dataOffset = offset + pos + 4;
      return "NConstantString";
    }
    // Compression byte, size of the lengths encoding, lengths, string bytes.
    const auto lengthsSize = readAt<uint32_t>(data, pos + 1);
    const int64_t blobPos = pos + 5 + lengthsSize;
    node.children[0] = parse(offset + pos + 5, lengthsSize).first;
    node.dataOffset = payloadOffset(data, offset, blobPos, DataType::String);
    result_.prepareNodes.push_back(nodeIdx);
    return "NString";
  }

  std::string_view encodingAt(int64_t offset, int64_t size) const {
    VELOX_CHECK_GE(size, EncodingPrefix::kFixedPrefixSize);
    VELOX_CHECK_LE(offset + size, stream_.size(), "Truncated Nimble stream");
    return stream_.substr(offset, size);
  }

  static NimbleNode makeNode(std::string_view data) {
    NimbleNode node{};
    node.encoding = static_cast<uint8_t>(EncodingPrefix::encodingType(data));
    node.dataType = static_cast<uint8_t>(EncodingPrefix::dataType(data));
    node.numRows = static_cast<int32_t>(
        readAt<uint32_t>(data, EncodingPrefix::kRowCountOffset));
    node.children[0] = -1;
    node.children[1] = -1;
    node.aux = -1;
    return node;
  }

  int32_t addNode(const NimbleNode& node) {
    result_.nodes.push_back(node);
    return static_cast<int32_t>(result_.nodes.size() - 1);
  }

  // Sets the fields of a Trivial, FixedBitWidth or Constant node.
  void setScalar(std::string_view data, int64_t offset, NimbleNode& node) {
    const int64_t pos = EncodingPrefix::kFixedPrefixSize;
    auto bits = physicalBits(EncodingPrefix::dataType(data));
    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (EncodingPrefix::encodingType(data)) {
      case EncodingType::Trivial:
        node.bitWidth = bits;
        node.dataOffset = payloadOffset(
            data, offset, pos + 1, EncodingPrefix::dataType(data));
        break;
      case EncodingType::FixedBitWidth:
        node.value = readPhysical(data, pos + 1, bits);
        node.bitWidth = readAt<uint8_t>(data, pos + 1 + bits / 8);
        node.dataOffset = payloadOffset(
            data, offset, pos + 2 + bits / 8, DataType::Undefined);
        break;
      case EncodingType::Constant:
        node.value = readPhysical(data, pos, bits);
        break;
      default:
        VELOX_NYI(
            "Nimble encoding of run lengths or positions not supported on GPU: {}",
            static_cast<int32_t>(EncodingPrefix::encodingType(data)));
    }
  }

  // Returns the offset of the payload of a leaf encoding at 'offset' in the
  // stream. The compression byte is at prefix end and the payload is from
  // 'payloadPos' to the end of 'data'. A compressed payload is decompressed
  // and appended to the decompressed area, which follows the stream.
  int64_t payloadOffset(
      std::string_view data,
      int64_t offset,
      int64_t payloadPos,
      DataType dataType) {
    auto compression = static_cast<CompressionType>(
        readAt<uint8_t>(data, EncodingPrefix::kFixedPrefixSize));
    if (compression == CompressionType::Uncompressed) {
      return offset + payloadPos;
    }
    VELOX_CHECK_NOT_NULL(
        pool_, "Compressed Nimble encoding needs a memory pool");
    auto uncompressed = nimble::Compression::uncompress(
        *pool_, compression, dataType, data.substr(payloadPos), nullptr);
    auto& area = result_.decompressed;
    area.resize(bits::roundUp(area.size(), sizeof(int64_t)));
    const auto position = static_cast<int64_t>(area.size());
    area.append(uncompressed->as<char>(), uncompressed->size());
    return result_.decompressedOffset() + position;
  }

  // Parses a Trivial, FixedBitWidth or Constant encoding of run lengths or
  // positions, which prepare ops read without a template accessor.
  int32_t parseScalar(int64_t offset, int64_t size) {
    auto data = encodingAt(offset, size);
    NimbleNode node = makeNode(data);
    setScalar(data, offset, node);
    return addNode(node);
  }

  // Parses a bool encoding of nulls or isCommon flags.
  int32_t parseBool(int64_t offset, int64_t size) {
    auto data = encodingAt(offset, size);
    VELOX_CHECK(
        EncodingPrefix::dataType(data) == DataType::Bool,
        "Nimble nulls must be bool");
    NimbleNode node = makeNode(data);
    const int32_t nodeIdx = addNode(node);
    const int64_t pos = EncodingPrefix::kFixedPrefixSize;
    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (EncodingPrefix::encodingType(data)) {
      case EncodingType::Trivial:
        node.bitWidth = 1;
        node.dataOffset =
            payloadOffset(data, offset, pos + 1, DataType::Undefined);
        break;
      case EncodingType::Constant:
        node.value = readAt<uint8_t>(data, pos);
        break;
      case EncodingType::SparseBool:
        node.value = readAt<uint8_t>(data, pos);
        node.children[0] = parseScalar(offset + pos + 1, size - pos - 1);
        break;
      case EncodingType::RLE: {
        auto lengthsSize = readAt<uint32_t>(data, pos);
        node.children[0] = parseScalar(offset + pos + 4, lengthsSize);
        node.value = readAt<uint8_t>(data, pos + 4 + lengthsSize);
        break;
      }
      default:
        VELOX_NYI(
            "Nimble bool encoding not supported on GPU: {}",
            static_cast<int32_t>(EncodingPrefix::encodingType(data)));
    }
    result_.nodes.at(nodeIdx) = node;
    return nodeIdx;
  }

 private:
  const std::string_view stream_;
  memory::MemoryPool* const pool_;
  NimbleStream& result_;
};

} // namespace

// static
NimbleStream NimbleStream::parse(
    std::string_view data,
    memory::MemoryPool* pool,
    bool isNulls) {
  NimbleStream result;
  result.streamSize = static_cast<int64_t>(data.size());
  TreeParser parser(data, pool, result);
  int64_t offset = 0;
  while (offset < data.size()) {
    const char* pos = data.data() + offset;
    VELOX_CHECK_LE(offset + nimble::kChunkHeaderSize, data.size());
    auto header = nimble::readChunkHeader(pos);
    VELOX_CHECK(
        header.compressionType == CompressionType::Uncompressed,
        "Compressed Nimble chunks are not supported on GPU: {}",
        toString(header.compressionType));
    offset += nimble::kChunkHeaderSize;
    int32_t root;
    std::string accessor;
    if (isNulls) {
      root = parser.parseBool(offset, header.length);
    } else {
      std::tie(root, accessor) = parser.parse(offset, header.length);
    }
    result.hasNulls |= isNulls ||
        result.nodes[root].encoding ==
            static_cast<uint8_t>(EncodingType::Nullable);
    int32_t shape = 0;
    if (!isNulls) {
      auto it =
          std::find(result.accessors.begin(), result.accessors.end(), accessor);
      if (it == result.accessors.end()) {
        it = result.accessors.insert(it, accessor);
      }
      shape = static_cast<int32_t>(it - result.accessors.begin());
    }
    result.chunks.push_back(
        ChunkDesc{
            .firstRow = result.numRows,
            .root = root,
            .shape = shape,
            .padding = 0});
    result.numRows += result.nodes[root].numRows;
    offset += header.length;
  }
  return result;
}

std::string NimbleStream::accessorList() const {
  std::string result;
  for (const auto& accessor : accessors) {
    if (!result.empty()) {
      result += ", ";
    }
    result += accessor;
  }
  return result;
}

} // namespace facebook::velox::wave
