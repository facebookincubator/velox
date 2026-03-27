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
  positions_.push_back(position);
}

void DeletionVectorWriter::addDeletedPositions(
    const std::vector<int64_t>& positions) {
  for (auto pos : positions) {
    addDeletedPosition(pos);
  }
}

std::string DeletionVectorWriter::serialize() const {
  if (positions_.empty()) {
    std::string data;
    appendLittleEndian(data, kSerialCookieNoRun);
    uint32_t zero = 0;
    appendLittleEndian(data, zero);
    return data;
  }

  // Sort and deduplicate positions.
  std::vector<int64_t> sorted = positions_;
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

  // Group positions by high 16 bits (container key).
  struct Container {
    uint16_t key;
    std::vector<uint16_t> values;
  };
  std::map<uint16_t, std::vector<uint16_t>> containerMap;
  for (auto pos : sorted) {
    auto key = static_cast<uint16_t>(pos >> 16);
    auto low = static_cast<uint16_t>(pos & 0xFFFF);
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
  for (auto& c : containers) {
    appendLittleEndian(data, c.key);
    auto cardMinus1 = static_cast<uint16_t>(c.values.size() - 1);
    appendLittleEndian(data, cardMinus1);
  }

  // Offset section (required for >= 4 containers).
  if (numContainers >= kNoOffsetThreshold) {
    // Calculate offsets: base = header + key/card pairs + offset section.
    uint32_t headerSize = 4 + 4 + numContainers * 4 + numContainers * 4;
    uint32_t runningOffset = headerSize;
    for (auto& c : containers) {
      appendLittleEndian(data, runningOffset);
      if (c.values.size() <= kMaxArrayContainerCardinality) {
        runningOffset += static_cast<uint32_t>(c.values.size()) * 2;
      } else {
        runningOffset += kBitsetBytes;
      }
    }
  }

  // Container data.
  for (auto& c : containers) {
    if (c.values.size() <= kMaxArrayContainerCardinality) {
      // Array container: sorted uint16 values.
      for (auto v : c.values) {
        appendLittleEndian(data, v);
      }
    } else {
      // Bitmap container: 8192 bytes (65536 bits).
      std::vector<uint64_t> bitmap(1'024, 0);
      for (auto v : c.values) {
        bitmap[v / 64] |= (1ULL << (v % 64));
      }
      for (auto word : bitmap) {
        appendLittleEndian(data, word);
      }
    }
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
  // Build the Puffin file in memory, then write atomically.
  //
  // Puffin file layout:
  //   [Magic "PUF1" (4 bytes)]
  //   [Blob data (N bytes)]
  //   [Footer]
  //   [Footer payload size (4 bytes, little-endian)]
  //   [Flags (4 bytes)]
  //   [Magic "PUF1" (4 bytes)]
  //
  // Footer is a JSON object with blob metadata.

  uint64_t blobOffset = kPuffinMagicSize;
  uint64_t blobLength = blobData.size();

  // Build footer JSON.
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

  // Assemble the file.
  std::string fileContent;
  // Header magic.
  fileContent.append(kPuffinMagic, kPuffinMagicSize);
  // Blob data.
  fileContent.append(blobData);
  // Footer JSON.
  fileContent.append(footerJson);
  // Footer payload size.
  uint32_t littleEndianSize = folly::Endian::little(footerPayloadSize);
  fileContent.append(
      reinterpret_cast<const char*>(&littleEndianSize),
      sizeof(littleEndianSize));
  // Flags.
  uint32_t littleEndianFlags = folly::Endian::little(kPuffinFooterFlags);
  fileContent.append(
      reinterpret_cast<const char*>(&littleEndianFlags),
      sizeof(littleEndianFlags));
  // Footer magic.
  fileContent.append(kPuffinMagic, kPuffinMagicSize);

  // Write to file.
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
