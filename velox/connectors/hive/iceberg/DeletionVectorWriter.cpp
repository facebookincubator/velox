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

#include <algorithm>
#include <cstring>

#include <folly/json.h>
#include <folly/lang/Bits.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

static constexpr uint32_t kSerialCookieNoRun = 12'346;
static constexpr uint32_t kNoOffsetThreshold = 4;
static constexpr uint32_t kMaxArrayContainerCardinality = 4'096;
static constexpr size_t kBitsetBytes = 8'192;

void appendLittleEndian(std::string& out, uint16_t val) {
  val = folly::Endian::little(val);
  out.append(reinterpret_cast<const char*>(&val), sizeof(val));
}

void appendLittleEndian(std::string& out, uint32_t val) {
  val = folly::Endian::little(val);
  out.append(reinterpret_cast<const char*>(&val), sizeof(val));
}

void appendLittleEndian(std::string& out, uint64_t val) {
  val = folly::Endian::little(val);
  out.append(reinterpret_cast<const char*>(&val), sizeof(val));
}

/// Puffin file magic: "PUF1" (4 bytes).
static constexpr char kPuffinMagic[] = {'\x50', '\x55', '\x46', '\x31'};
static constexpr size_t kPuffinMagicSize = 4;

/// Puffin footer flags: no compression, no encryption.
static constexpr uint32_t kPuffinFooterFlags = 0;

} // namespace

void DeletionVectorWriter::addDeletedPosition(int64_t position) {
  VELOX_CHECK_GE(position, 0, "Deleted position must be non-negative.");
  VELOX_CHECK_LE(
      position,
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
      "Deleted position exceeds uint32 range for Roaring Bitmap serialization.");
  positions_.push_back(position);
}

void DeletionVectorWriter::addDeletedPositions(
    const std::vector<int64_t>& positions) {
  for (auto pos : positions) {
    addDeletedPosition(pos);
  }
}

std::string DeletionVectorWriter::serialize32(
    const std::vector<uint32_t>& sorted) const {
  if (sorted.empty()) {
    std::string data;
    appendLittleEndian(data, kSerialCookieNoRun);
    uint32_t zero = 0;
    appendLittleEndian(data, zero);
    return data;
  }

  // Group values by high 16 bits (container key).
  struct Container {
    uint16_t key;
    std::vector<uint16_t> values;
  };
  std::map<uint16_t, std::vector<uint16_t>> containerMap;
  for (auto val : sorted) {
    auto key = static_cast<uint16_t>(val >> 16);
    auto low = static_cast<uint16_t>(val & 0xFFFF);
    containerMap[key].push_back(low);
  }

  std::vector<Container> containers;
  containers.reserve(containerMap.size());
  for (auto& [key, vals] : containerMap) {
    containers.push_back({key, std::move(vals)});
  }

  uint32_t numContainers = static_cast<uint32_t>(containers.size());

  std::string data;

  // Header: cookie + container count.
  appendLittleEndian(data, kSerialCookieNoRun);
  appendLittleEndian(data, numContainers);

  // Key-cardinality pairs.
  for (auto& container : containers) {
    appendLittleEndian(data, container.key);
    auto cardMinus1 = static_cast<uint16_t>(container.values.size() - 1);
    appendLittleEndian(data, cardMinus1);
  }

  // Offset section (required for >= 4 containers).
  if (numContainers >= kNoOffsetThreshold) {
    uint32_t headerSize = 4 + 4 + numContainers * 4 + numContainers * 4;
    uint32_t runningOffset = headerSize;
    for (auto& container : containers) {
      appendLittleEndian(data, runningOffset);
      if (container.values.size() <= kMaxArrayContainerCardinality) {
        runningOffset += static_cast<uint32_t>(container.values.size()) * 2;
      } else {
        runningOffset += kBitsetBytes;
      }
    }
  }

  // Container data.
  for (auto& container : containers) {
    if (container.values.size() <= kMaxArrayContainerCardinality) {
      for (auto value : container.values) {
        appendLittleEndian(data, value);
      }
    } else {
      std::vector<uint64_t> bitmap(1'024, 0);
      for (auto value : container.values) {
        bitmap[value / 64] |= (1ULL << (value % 64));
      }
      for (auto word : bitmap) {
        appendLittleEndian(data, word);
      }
    }
  }

  return data;
}

std::string DeletionVectorWriter::serialize() const {
  if (positions_.empty()) {
    // Empty 64-bit bitmap: 0 groups.
    std::string data;
    appendLittleEndian(data, static_cast<uint64_t>(0));
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
    auto highBits = static_cast<uint32_t>(static_cast<uint64_t>(pos) >> 32);
    auto lowBits =
        static_cast<uint32_t>(static_cast<uint64_t>(pos) & 0xFFFFFFFF);
    groups[highBits].push_back(lowBits);
  }

  std::string data;
  appendLittleEndian(data, static_cast<uint64_t>(groups.size()));

  for (auto& [highBits, lowValues] : groups) {
    appendLittleEndian(data, highBits);
    auto bitmap32 = serialize32(lowValues);
    data.append(bitmap32);
  }

  return data;
}

void DeletionVectorWriter::clear() {
  positions_.clear();
}

std::pair<uint64_t, uint64_t> writePuffinFile(
    const std::string& filePath,
    const std::string& blobData,
    const std::string& referencedDataFile) {
  uint64_t blobOffset = kPuffinMagicSize;
  uint64_t blobLength = blobData.size();

  folly::dynamic blobMeta =
      folly::dynamic::object("type", "deletion-vector-v1")(
          "fields",
          folly::dynamic::array(
              folly::dynamic::object("source-field-id", 2147483646)));
  blobMeta["offset"] = blobOffset;
  blobMeta["length"] = blobLength;
  blobMeta["compression-codec"] = "none";

  folly::dynamic properties = folly::dynamic::object;
  properties["referenced-data-file"] = referencedDataFile;
  blobMeta["properties"] = properties;

  folly::dynamic footer = folly::dynamic::object;
  footer["blobs"] = folly::dynamic::array(blobMeta);
  footer["properties"] = folly::dynamic::object;

  std::string footerJson = folly::toJson(footer);
  uint32_t footerPayloadSize = static_cast<uint32_t>(footerJson.size());

  std::string fileContent;
  fileContent.append(kPuffinMagic, kPuffinMagicSize);
  fileContent.append(blobData);
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

  std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
  VELOX_CHECK(
      out.good(), "Failed to open Puffin file for writing: {}", filePath);
  out.write(
      fileContent.data(), static_cast<std::streamsize>(fileContent.size()));
  out.close();
  VELOX_CHECK(!out.fail(), "Failed to write Puffin file: {}", filePath);

  return {blobOffset, blobLength};
}

} // namespace facebook::velox::connector::hive::iceberg
