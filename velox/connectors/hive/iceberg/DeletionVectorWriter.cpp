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
#include <cstring>

#include <folly/json.h>
#include <folly/lang/Bits.h>
#include <zlib.h>

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
constexpr uint32_t kPuffinFooterFlags = 0;

// Puffin blob metadata constants (per Iceberg V3 deletion vector spec).
// The blob's "fields" list names the row-position metadata column
// (MetadataColumns.ROW_POSITION, INT_MAX - 2), matching Iceberg's own
// BaseDVFileWriter. Note this is not the positional-delete file's "pos"
// column (INT_MAX - 102), which IcebergMetadataColumns uses for a different
// purpose.
constexpr int32_t kRowPositionFieldId = 2'147'483'645;
// A deletion vector is written before its snapshot is assigned, so Iceberg
// records -1 for both. The fields are required by readers regardless.
constexpr int64_t kUnassignedSnapshotId = -1;
constexpr int64_t kUnassignedSequenceNumber = -1;

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

  // Iceberg stores the standard CRC-32 (java.util.zip.CRC32) over magic +
  // bitmap. zlib's crc32 is that same finalized IEEE 802.3 CRC-32.
  uLong crcState = crc32(0L, Z_NULL, 0);
  crcState = crc32(
      crcState,
      reinterpret_cast<const Bytef*>(magicAndVector.data()),
      static_cast<uInt>(magicAndVector.size()));
  const auto crc = static_cast<uint32_t>(crcState);

  std::string framed;
  framed.reserve(sizeof(uint32_t) + magicAndVector.size() + sizeof(uint32_t));
  writeBigEndian(framed, static_cast<uint32_t>(magicAndVector.size()));
  framed.append(magicAndVector);
  writeBigEndian(framed, crc);
  return framed;
}

// A run of sorted positions [begin, end) that share the same high 48 bits,
// i.e. one roaring container.
struct ContainerRange {
  size_t begin;
  size_t end;

  size_t cardinality() const {
    return end - begin;
  }
};

uint32_t containerBytes(const ContainerRange& container) {
  return container.cardinality() <= kMaxArrayContainerCardinality
      ? static_cast<uint32_t>(container.cardinality() * sizeof(uint16_t))
      : static_cast<uint32_t>(kBitmapContainerBytes);
}

// Appends one 32-bit portable roaring bitmap for the sorted, distinct
// 'positions' in [begin, end), which all share the same high 32 bits. Each
// container is written straight from its run of positions.
void appendRoaring32(
    std::string& out,
    const std::vector<int64_t>& positions,
    size_t begin,
    size_t end,
    std::vector<ContainerRange>& containers) {
  containers.clear();
  for (size_t i = begin; i < end;) {
    const auto key = static_cast<uint32_t>(positions[i]) >> 16;
    size_t containerEnd = i + 1;
    while (containerEnd < end &&
           (static_cast<uint32_t>(positions[containerEnd]) >> 16) == key) {
      ++containerEnd;
    }
    containers.push_back({i, containerEnd});
    i = containerEnd;
  }

  const auto numContainers = static_cast<uint32_t>(containers.size());
  writeLittleEndian(out, kSerialCookieNoRun);
  writeLittleEndian(out, numContainers);
  for (const auto& container : containers) {
    writeLittleEndian(
        out,
        static_cast<uint16_t>(
            static_cast<uint32_t>(positions[container.begin]) >> 16));
    writeLittleEndian(out, static_cast<uint16_t>(container.cardinality() - 1));
  }
  // Offsets are measured from the start of this 32-bit bitmap: cookie (4) +
  // count (4) + key/cardinality pairs (4 each) + offsets (4 each).
  uint32_t offset = 4 + 4 + numContainers * 4 + numContainers * 4;
  for (const auto& container : containers) {
    writeLittleEndian(out, offset);
    offset += containerBytes(container);
  }

  for (const auto& container : containers) {
    const size_t dataStart = out.size();
    out.resize(dataStart + containerBytes(container), '\0');
    char* data = out.data() + dataStart;
    if (container.cardinality() <= kMaxArrayContainerCardinality) {
      for (size_t i = container.begin; i < container.end; ++i) {
        const uint16_t value =
            folly::Endian::little(static_cast<uint16_t>(positions[i]));
        std::memcpy(data, &value, sizeof(value));
        data += sizeof(value);
      }
    } else {
      uint64_t words[kBitmapContainerWords] = {};
      for (size_t i = container.begin; i < container.end; ++i) {
        const auto value = static_cast<uint16_t>(positions[i]);
        words[value / 64] |= 1ULL << (value % 64);
      }
      for (auto& word : words) {
        word = folly::Endian::little(word);
      }
      std::memcpy(data, words, sizeof(words));
    }
  }
}

} // namespace

void DeletionVectorWriter::addDeletedPosition(int64_t position) {
  VELOX_CHECK_GE(position, 0, "Deleted position must be non-negative.");
  VELOX_CHECK_LE(
      position,
      kMaxPosition,
      "Deleted position exceeds the maximum the Iceberg deletion-vector "
      "format can represent: {}",
      position);
  if (sortedPrefix_ == positions_.size() &&
      (positions_.empty() || position > positions_.back())) {
    ++sortedPrefix_;
  }
  positions_.push_back(position);
}

void DeletionVectorWriter::addDeletedPositions(
    const std::vector<int64_t>& positions) {
  for (auto pos : positions) {
    addDeletedPosition(pos);
  }
}

void DeletionVectorWriter::normalize() {
  if (sortedPrefix_ == positions_.size()) {
    return;
  }
  const auto middle = positions_.begin() + sortedPrefix_;
  if (!std::is_sorted(middle, positions_.end())) {
    std::sort(middle, positions_.end());
  }
  std::inplace_merge(positions_.begin(), middle, positions_.end());
  positions_.erase(
      std::unique(positions_.begin(), positions_.end()), positions_.end());
  sortedPrefix_ = positions_.size();
}

size_t DeletionVectorWriter::numDistinctPositions() {
  normalize();
  return positions_.size();
}

std::string DeletionVectorWriter::serialize() {
  normalize();
  const auto& sorted = positions_;

  // Roaring64Bitmap portable format: [numGroups: uint64] then for each group
  // of positions sharing the high 32 bits: [highBits: uint32] [32-bit
  // RoaringBitmap of the low 32 bits].
  uint64_t numGroups = 0;
  for (size_t i = 0; i < sorted.size(); ++i) {
    numGroups += i == 0 || (sorted[i] >> 32) != (sorted[i - 1] >> 32);
  }

  std::string data;
  writeLittleEndian(data, numGroups);
  std::vector<ContainerRange> containers;
  for (size_t i = 0; i < sorted.size();) {
    // Safe cast: addDeletedPosition() rejects negative values.
    const auto highBits = static_cast<uint32_t>(sorted[i] >> 32);
    size_t groupEnd = i + 1;
    while (groupEnd < sorted.size() &&
           static_cast<uint32_t>(sorted[groupEnd] >> 32) == highBits) {
      ++groupEnd;
    }
    writeLittleEndian(data, highBits);
    appendRoaring32(data, sorted, i, groupEnd, containers);
    i = groupEnd;
  }
  return data;
}

void DeletionVectorWriter::clear() {
  positions_.clear();
  sortedPrefix_ = 0;
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

  folly::dynamic blobMeta =
      folly::dynamic::object("type", kDeletionVectorBlobType)(
          "fields", folly::dynamic::array(kRowPositionFieldId));
  blobMeta["snapshot-id"] = kUnassignedSnapshotId;
  blobMeta["sequence-number"] = kUnassignedSequenceNumber;
  blobMeta["offset"] = blobOffset;
  blobMeta["length"] = blobLength;
  // Uncompressed blobs omit "compression-codec": Iceberg's FileMetadataParser
  // writes the field only for a non-null codec, and PuffinCompressionCodec
  // .forName() has no "none" entry, so emitting it literally would make a
  // spec-compliant reader throw.

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
