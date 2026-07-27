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

#include "velox/connectors/hive/iceberg/DeletionVectorWriter.h"

#include "velox/connectors/hive/iceberg/DeletionVectorFormat.h"

#include <algorithm>

#include <folly/hash/Checksum.h>
#include <folly/json.h>
#include <folly/lang/Bits.h>

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/DataBuffer.h"
#include "velox/dwio/common/FileSink.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Roaring Bitmap portable format constants.
constexpr uint32_t kSerialCookieNoRun = 12'346;
constexpr uint32_t kMaxArrayContainerCardinality = 4'096;
// Full bitmap container: 2^16 bits = 1024 uint64 words = 8192 bytes.
constexpr size_t kBitmapContainerBytes = 8'192;
constexpr size_t kBitmapContainerWords = 1'024;

// Puffin file format constants (per Iceberg spec). Magic is "PFA1".
constexpr char kPuffinMagic[] = {'\x50', '\x46', '\x41', '\x31'};
constexpr size_t kPuffinMagicSize = 4;
constexpr uint32_t kPuffinFooterFlags = 0;

// Puffin blob metadata constants (per Iceberg V3 deletion vector spec).
constexpr char kDeletionVectorBlobType[] = "deletion-vector-v1";
constexpr char kCompressionCodecNone[] = "none";
// Iceberg spec: source-field-id for whole-row deletes is INT_MAX - 1.
constexpr int32_t kWholeRowDeleteFieldId = 2'147'483'646;

void writeLittleEndian(std::string& out, uint16_t val) {
  val = folly::Endian::little(val);
  out.append(reinterpret_cast<const char*>(&val), sizeof(val));
}

void writeLittleEndian(std::string& out, uint32_t val) {
  val = folly::Endian::little(val);
  out.append(reinterpret_cast<const char*>(&val), sizeof(val));
}

void writeLittleEndian(std::string& out, uint64_t val) {
  val = folly::Endian::little(val);
  out.append(reinterpret_cast<const char*>(&val), sizeof(val));
}

void writeBigEndian(std::string& out, uint32_t val) {
  val = folly::Endian::big(val);
  out.append(reinterpret_cast<const char*>(&val), sizeof(val));
}

// Wraps the serialized roaring bitmap in the Iceberg deletion-vector-v1 blob
// frame: [length: 4B BE][magic: D1 D3 39 64][bitmap][CRC-32: 4B BE], where
// 'length' and the CRC-32 cover magic + bitmap. This is the on-disk layout
// Iceberg's BitmapPositionDeleteIndex produces, so the blob is readable by
// spec-compliant Iceberg engines and vice versa.
std::string frameDeletionVector(const std::string& bitmap) {
  std::string magicAndVector;
  magicAndVector.reserve(kDeletionVectorMagicSize + bitmap.size());
  magicAndVector.append(kDeletionVectorMagic, kDeletionVectorMagicSize);
  magicAndVector.append(bitmap);

  const uint32_t crc = folly::crc32(
      reinterpret_cast<const uint8_t*>(magicAndVector.data()),
      magicAndVector.size());

  std::string framed;
  framed.reserve(sizeof(uint32_t) + magicAndVector.size() + sizeof(uint32_t));
  writeBigEndian(framed, static_cast<uint32_t>(magicAndVector.size()));
  framed.append(magicAndVector);
  writeBigEndian(framed, crc);
  return framed;
}

// Serializes the key-cardinality header for a 32-bit Roaring Bitmap.
void serializeKeyCardinality(
    std::string& data,
    const std::vector<std::pair<uint16_t, std::vector<uint16_t>>>& containers) {
  for (const auto& [key, values] : containers) {
    writeLittleEndian(data, key);
    auto cardMinus1 = static_cast<uint16_t>(values.size() - 1);
    writeLittleEndian(data, cardMinus1);
  }
}

// Serializes the offset section for a 32-bit Roaring Bitmap.
// Offsets point to where each container's data begins, measured from the start
// of the serialized bitmap. The header consists of: cookie (4) + count (4) +
// key-cardinality pairs (numContainers * 4) + offset section (numContainers *
// 4). Offsets are relative to byte 0 of the serialized output, so the first
// container's data starts immediately after the full header including offsets.
void serializeOffsets(
    std::string& data,
    const std::vector<std::pair<uint16_t, std::vector<uint16_t>>>& containers) {
  auto numContainers = static_cast<uint32_t>(containers.size());
  uint32_t headerSize = 4 + 4 + numContainers * 4 + numContainers * 4;
  uint32_t runningOffset = headerSize;
  for (const auto& [key, values] : containers) {
    writeLittleEndian(data, runningOffset);
    if (values.size() <= kMaxArrayContainerCardinality) {
      runningOffset += static_cast<uint32_t>(values.size()) * 2;
    } else {
      runningOffset += kBitmapContainerBytes;
    }
  }
}

// Serializes container data (array or bitmap) for a 32-bit Roaring Bitmap.
// Array containers store sorted uint16 values directly. Bitmap containers
// store a 65536-bit bitset as 1024 little-endian uint64 words, covering the
// full uint16 range [0, 65535].
void serializeContainerData(
    std::string& data,
    const std::vector<std::pair<uint16_t, std::vector<uint16_t>>>& containers) {
  for (const auto& [key, values] : containers) {
    if (values.size() <= kMaxArrayContainerCardinality) {
      for (auto value : values) {
        writeLittleEndian(data, value);
      }
    } else {
      std::vector<uint64_t> bitmap(kBitmapContainerWords, 0);
      for (auto value : values) {
        bitmap[value / 64] |= (1ULL << (value % 64));
      }
      for (auto word : bitmap) {
        writeLittleEndian(data, word);
      }
    }
  }
}

} // namespace

void DeletionVectorWriter::addDeletedPosition(int64_t position) {
  VELOX_CHECK_GE(position, 0, "Deleted position must be non-negative.");
  positions_.push_back(position);
}

void DeletionVectorWriter::addDeletedPositions(
    const std::vector<int64_t>& positions) {
  for (auto pos : positions) {
    addDeletedPosition(pos);
  }
}

size_t DeletionVectorWriter::numDistinctPositions() const {
  std::vector<int64_t> sorted = positions_;
  std::sort(sorted.begin(), sorted.end());
  return static_cast<size_t>(
      std::unique(sorted.begin(), sorted.end()) - sorted.begin());
}

std::string DeletionVectorWriter::serialize32(
    const std::vector<uint32_t>& sorted) const {
  if (sorted.empty()) {
    std::string data;
    writeLittleEndian(data, kSerialCookieNoRun);
    uint32_t zero = 0;
    writeLittleEndian(data, zero);
    return data;
  }

  // Group values by high 16 bits (container key).
  std::map<uint16_t, std::vector<uint16_t>> containerMap;
  for (auto val : sorted) {
    auto key = static_cast<uint16_t>(val >> 16);
    auto low = static_cast<uint16_t>(val & 0xFFFF);
    containerMap[key].push_back(low);
  }

  std::vector<std::pair<uint16_t, std::vector<uint16_t>>> containers(
      containerMap.begin(), containerMap.end());
  auto numContainers = static_cast<uint32_t>(containers.size());

  std::string data;
  writeLittleEndian(data, kSerialCookieNoRun);
  writeLittleEndian(data, numContainers);
  serializeKeyCardinality(data, containers);
  serializeOffsets(data, containers);
  serializeContainerData(data, containers);
  return data;
}

std::string DeletionVectorWriter::serialize() const {
  if (positions_.empty()) {
    std::string data;
    writeLittleEndian(data, static_cast<uint64_t>(0));
    return data;
  }

  // Sort and deduplicate positions.
  std::vector<int64_t> sorted = positions_;
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

  // Partition into 32-bit high groups.
  // Roaring64Bitmap format: [numGroups: uint64] then for each group:
  //   [highBits: uint32] [serialized 32-bit RoaringBitmap]
  std::map<uint32_t, std::vector<uint32_t>> groups;
  for (auto pos : sorted) {
    // Safe cast: addDeletedPosition() rejects negative values.
    auto upos = static_cast<uint64_t>(pos);
    groups[static_cast<uint32_t>(upos >> 32)].push_back(
        static_cast<uint32_t>(upos & 0xFFFFFFFF));
  }

  std::string data;
  writeLittleEndian(data, static_cast<uint64_t>(groups.size()));

  for (auto& [highBits, lowValues] : groups) {
    writeLittleEndian(data, highBits);
    data.append(serialize32(lowValues));
  }

  return data;
}

void DeletionVectorWriter::clear() {
  positions_.clear();
}

std::pair<uint64_t, uint64_t> writePuffinFile(
    dwio::common::FileSink& sink,
    memory::MemoryPool& pool,
    const std::string& blobData,
    const std::string& referencedDataFile,
    int64_t cardinality) {
  // Wrap the raw roaring bitmap in the Iceberg deletion-vector-v1 frame so the
  // blob is interoperable with spec-compliant Iceberg readers.
  const std::string framedBlob = frameDeletionVector(blobData);
  uint64_t blobOffset = kPuffinMagicSize;
  uint64_t blobLength = framedBlob.size();

  folly::dynamic blobMeta = folly::dynamic::object(
      "type", kDeletionVectorBlobType)(
      "fields",
      folly::dynamic::array(
          folly::dynamic::object("source-field-id", kWholeRowDeleteFieldId)));
  blobMeta["offset"] = blobOffset;
  blobMeta["length"] = blobLength;
  blobMeta["compression-codec"] = kCompressionCodecNone;

  folly::dynamic properties = folly::dynamic::object;
  properties["referenced-data-file"] = referencedDataFile;
  // Iceberg V3 requires the deletion-vector-v1 blob to carry its cardinality
  // (the number of deleted positions).
  properties["cardinality"] = std::to_string(cardinality);
  blobMeta["properties"] = properties;

  folly::dynamic footer = folly::dynamic::object;
  footer["blobs"] = folly::dynamic::array(blobMeta);
  footer["properties"] = folly::dynamic::object;

  std::string footerJson = folly::toJson(footer);
  uint32_t footerPayloadSize = static_cast<uint32_t>(footerJson.size());

  std::string fileContent;
  fileContent.append(kPuffinMagic, kPuffinMagicSize);
  fileContent.append(framedBlob);
  // Footer = Magic FooterPayload FooterPayloadSize Flags Magic (Puffin spec).
  fileContent.append(kPuffinMagic, kPuffinMagicSize);
  fileContent.append(footerJson);
  uint32_t littleEndianSize = folly::Endian::little(footerPayloadSize);
  fileContent.append(
      reinterpret_cast<const char*>(&littleEndianSize),
      sizeof(littleEndianSize));
  uint32_t littleEndianFlags = folly::Endian::little(kPuffinFooterFlags);
  fileContent.append(
      reinterpret_cast<const char*>(&littleEndianFlags),
      sizeof(littleEndianFlags));
  fileContent.append(kPuffinMagic, kPuffinMagicSize);

  // Stage the serialized puffin bytes into a DataBuffer<char> and write
  // them through the registered FileSink. Going through the sink lets the
  // bytes land on whatever filesystem (local, warm storage, S3, ...) has a
  // FileSink factory registered for the destination URI scheme.
  // DataBuffer::append reserves and copies the bytes internally, so the
  // staging copy stays inside the bounds-checked buffer API.
  dwio::common::DataBuffer<char> buffer(pool);
  buffer.append(0, fileContent.data(), fileContent.size());
  sink.write(std::move(buffer));

  return {blobOffset, blobLength};
}

} // namespace facebook::velox::connector::hive::iceberg
