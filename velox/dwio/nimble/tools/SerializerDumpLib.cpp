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

#include "velox/dwio/nimble/tools/SerializerDumpLib.h"

#include <fmt/format.h>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/serializer/legacy/TrailerReader.h"

namespace facebook::nimble::tools {

namespace {

uint32_t dataTypeElementSize(DataType dataType) {
  switch (dataType) {
    case DataType::Int8:
    case DataType::Uint8:
    case DataType::Bool:
      return 1;
    case DataType::Int16:
    case DataType::Uint16:
      return 2;
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Float:
      return 4;
    case DataType::Int64:
    case DataType::Uint64:
    case DataType::Double:
      return 8;
    case DataType::String:
    case DataType::Undefined:
      return 0;
    default:
      NIMBLE_UNSUPPORTED(
          "Unsupported data type: {}", static_cast<int>(dataType));
  }
}

} // namespace

SerializationDump::SerializationStats SerializationDump::serializationStats(
    std::string_view serialized) {
  SerializationStats info{};

  NIMBLE_USER_CHECK_GE(serialized.size(), 2, "Serialized buffer too small");

  const char* pos = serialized.data();
  const char* end = pos + serialized.size();

  const auto header = serde::readSerializationHeader(pos, end, true);
  info.version = header.version;
  info.rowCount = header.rowCount;

  // Walk the sparse (indices, sizes) trailer directly. Emit "empty" entries
  // for gaps between non-zero slots so the offset-indexed view is preserved.
  // Legacy kLegacyCompact blobs are decoded via the frozen legacy reader.
  auto [indices, sizes] = usesLegacyTrailer(info.version)
      ? serde::legacy::readLegacyTrailerStreamMetadata(end)
      // NOLINTNEXTLINE(facebook-hte-DetailCall)
      : serde::detail::readTrailerStreamMetadata(end);
  const uint32_t numStreams = indices.empty() ? 0 : indices.back() + 1;
  info.streams.reserve(numStreams);

  size_t k = 0;
  for (uint32_t i = 0; i < numStreams; ++i) {
    StreamStats si{};
    si.streamIndex = i;

    auto label = streamLabels_.streamLabel(static_cast<offset_size>(i));
    if (!label.empty()) {
      si.schemaLabel = std::string(label);
    }

    if (k >= indices.size() || indices[k] != i) {
      si.empty = true;
      info.streams.push_back(std::move(si));
      continue;
    }
    const uint32_t streamSize = sizes[k++];

    si.encodingType = static_cast<EncodingType>(static_cast<uint8_t>(pos[0]));
    si.dataType = static_cast<DataType>(static_cast<uint8_t>(pos[1]));

    // Read the row count from the encoding prefix (varint at offset 2).
    const char* rowCountPos = pos + 2;
    si.rowCount = varint::readVarint32(&rowCountPos);

    // Read compression type byte for TrivialEncoding only.
    // TrivialEncoding layout:
    //   [EncodingType][DataType][rowCount:varint][CompressionType:1B][data]
    // Other encodings (Varint, RLE, FixedBitWidth, etc.) have encoding-specific
    // prefix bytes at this position, so we only parse compression type for
    // Trivial.
    if (si.encodingType == EncodingType::Trivial &&
        rowCountPos < pos + streamSize) {
      si.compressionType =
          static_cast<CompressionType>(static_cast<uint8_t>(*rowCountPos));
    }

    uint32_t elemSize = dataTypeElementSize(si.dataType);
    si.rawSize = static_cast<uint64_t>(si.rowCount) * elemSize;
    si.encodedSize = streamSize;

    info.rawSize += si.rawSize;
    info.encodedSize += si.encodedSize;

    info.streams.push_back(std::move(si));
    pos += streamSize;
  }

  return info;
}

std::string SerializationDump::SerializationStats::toString() const {
  std::string result = fmt::format(
      "Nimble encoding dump: version={}, rowCount={}, numStreams={}\n",
      static_cast<int>(version),
      rowCount,
      streams.size());

  for (const auto& si : streams) {
    if (si.empty) {
      result += fmt::format(
          "  stream[{}]: empty (size=0){}\n",
          si.streamIndex,
          si.schemaLabel.empty() ? ""
                                 : fmt::format(", schema={}", si.schemaLabel));
      continue;
    }

    double ratio =
        si.rawSize > 0 ? static_cast<double>(si.encodedSize) / si.rawSize : 0;
    result += fmt::format(
        "  stream[{}]: encoding={}, dataType={}, compression={}, rows={}, raw={} bytes,"
        " encoded={} bytes, ratio={:.2f}{}\n",
        si.streamIndex,
        nimble::toString(si.encodingType),
        nimble::toString(si.dataType),
        nimble::toString(si.compressionType),
        si.rowCount,
        si.rawSize,
        si.encodedSize,
        ratio,
        si.schemaLabel.empty() ? ""
                               : fmt::format(", schema={}", si.schemaLabel));
  }

  double overallRatio =
      rawSize > 0 ? static_cast<double>(encodedSize) / rawSize : 0;
  result += fmt::format(
      "Totals: raw={} bytes, encoded={} bytes, ratio={:.2f}",
      rawSize,
      encodedSize,
      overallRatio);

  return result;
}

SerializationDump::SerializationDump(
    std::shared_ptr<const Type> schema,
    velox::memory::MemoryPool* pool)
    : pool_{pool}, streamLabels_{schema} {}

void SerializationDump::addSerialization(std::string_view serialized) {
  auto stats = serializationStats(serialized);

  if (stats_.streams.empty()) {
    stats_ = std::move(stats);
  } else {
    // Stream count can vary across serializations because flat map keys
    // differ per row, producing different numbers of streams. Grow the
    // accumulator when a new serialization has more streams, copying metadata
    // (labels, encoding info) from the new stats for the new entries.
    const auto prevSize = stats_.streams.size();
    if (stats.streams.size() > prevSize) {
      stats_.streams.resize(stats.streams.size());
      for (size_t i = prevSize; i < stats.streams.size(); ++i) {
        stats_.streams[i] = stats.streams[i];
      }
    }
    // Accumulate per-stream sizes for streams that existed before the resize.
    for (size_t i = 0; i < std::min(prevSize, stats.streams.size()); ++i) {
      stats_.streams[i].rawSize += stats.streams[i].rawSize;
      stats_.streams[i].encodedSize += stats.streams[i].encodedSize;
      stats_.streams[i].rowCount += stats.streams[i].rowCount;
    }
    stats_.rawSize += stats.rawSize;
    stats_.encodedSize += stats.encodedSize;
    stats_.rowCount += stats.rowCount;
  }
}

void SerializationDump::addSerializations(
    const std::vector<std::string_view>& serializations) {
  for (const auto& serialized : serializations) {
    addSerialization(serialized);
  }
}

void SerializationDump::reset() {
  stats_ = {};
}

std::string SerializationDump::toString() const {
  return stats_.toString();
}

} // namespace facebook::nimble::tools
