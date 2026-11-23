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

#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"

#include "velox/connectors/hive/iceberg/DeletionVectorFormat.h"

#include <folly/json.h>
#include <folly/lang/Bits.h>
#include <zlib.h>

#include <cstring>
#include <string_view>

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {
static constexpr uint32_t kSerialCookieNoRun = 12'346;
static constexpr uint32_t kSerialCookie = 12'347;

uint32_t readBigEndian32(const char* data) {
  uint32_t value;
  std::memcpy(&value, data, sizeof(value));
  return folly::Endian::big(value);
}

uint32_t readLittleEndian32(const char* data) {
  uint32_t value;
  std::memcpy(&value, data, sizeof(value));
  return folly::Endian::little(value);
}

// Byte range of a blob inside a Puffin file.
struct BlobLocation {
  uint64_t offset;
  uint64_t length;
};

// Reads the Puffin magic that both opens the file and brackets the footer.
bool startsWithPuffinMagic(ReadFile& file, const FileIoContext& ioContext) {
  if (file.size() < kPuffinMagicSize) {
    return false;
  }
  std::string magic(kPuffinMagicSize, '\0');
  file.pread(0, kPuffinMagicSize, magic.data(), ioContext);
  return std::string_view(magic) ==
      std::string_view(kPuffinMagic, kPuffinMagicSize);
}

// Locates the deletion-vector blob by parsing the Puffin footer, for files
// whose manifest entry carried no blob offset or length. The footer is:
//   Magic FooterPayload FooterPayloadSize Flags Magic
// with FooterPayloadSize and Flags each 4 bytes little-endian, read backwards
// from end of file. 'referencedDataFile' disambiguates when the file holds
// vectors for several data files; an empty value requires a single candidate.
BlobLocation locateBlobFromPuffinFooter(
    ReadFile& file,
    const std::string& referencedDataFile,
    const FileIoContext& ioContext) {
  // FooterPayloadSize + Flags + trailing Magic.
  constexpr uint64_t kTrailerSize = 4 + 4 + kPuffinMagicSize;
  // Leading Magic and the Magic that opens the footer bracket any payload.
  constexpr uint64_t kMinFileSize = kTrailerSize + 2 * kPuffinMagicSize;

  const uint64_t fileSize = file.size();
  VELOX_CHECK_GE(
      fileSize,
      kMinFileSize,
      "Puffin file is too small to contain a footer: {} bytes.",
      fileSize);

  std::string trailer(kTrailerSize, '\0');
  file.pread(fileSize - kTrailerSize, kTrailerSize, trailer.data(), ioContext);
  const std::string_view puffinMagic(kPuffinMagic, kPuffinMagicSize);
  VELOX_CHECK_EQ(
      std::string_view(trailer).substr(8),
      puffinMagic,
      "Puffin file does not end with the expected magic.");

  const uint32_t flags = readLittleEndian32(trailer.data() + 4);
  // Bit 0 of the first flag byte marks a compressed footer payload. Deletion
  // vectors are always written uncompressed, so this is unreachable for files
  // we produce and unsupported for files we do not.
  VELOX_CHECK_EQ(
      flags & 1u, 0u, "Compressed Puffin footer payloads are not supported.");

  // payloadOffset must leave room for the leading magic and the magic that
  // opens the footer, so payloadSize + kMinFileSize must fit within the file.
  const uint64_t payloadSize = readLittleEndian32(trailer.data());
  VELOX_CHECK_LE(
      payloadSize + kMinFileSize,
      fileSize,
      "Puffin footer payload size {} does not fit in a {}-byte file.",
      payloadSize,
      fileSize);

  const uint64_t payloadOffset = fileSize - kTrailerSize - payloadSize;
  std::string footerMagic(kPuffinMagicSize, '\0');
  file.pread(
      payloadOffset - kPuffinMagicSize,
      kPuffinMagicSize,
      footerMagic.data(),
      ioContext);
  VELOX_CHECK_EQ(
      std::string_view(footerMagic),
      puffinMagic,
      "Puffin footer payload is not preceded by the expected magic.");

  std::string payload(payloadSize, '\0');
  file.pread(payloadOffset, payloadSize, payload.data(), ioContext);

  folly::dynamic footer;
  try {
    footer = folly::parseJson(payload);
  } catch (const std::exception& e) {
    VELOX_FAIL("Failed to parse Puffin footer payload: {}", e.what());
  }

  const auto* blobs = footer.get_ptr("blobs");
  VELOX_CHECK(
      blobs != nullptr && blobs->isArray(),
      "Puffin footer has no \"blobs\" array.");

  std::optional<BlobLocation> found;
  for (const auto& blob : *blobs) {
    const auto* type = blob.get_ptr("type");
    if (type == nullptr || !type->isString() ||
        type->asString() != kDeletionVectorBlobType) {
      continue;
    }
    if (!referencedDataFile.empty()) {
      const auto* properties = blob.get_ptr("properties");
      const auto* referenced = properties == nullptr
          ? nullptr
          : properties->get_ptr("referenced-data-file");
      if (referenced == nullptr || !referenced->isString() ||
          referenced->asString() != referencedDataFile) {
        continue;
      }
    }
    const auto* offset = blob.get_ptr("offset");
    const auto* length = blob.get_ptr("length");
    VELOX_CHECK(
        offset != nullptr && offset->isInt() && length != nullptr &&
            length->isInt(),
        "Puffin blob metadata is missing a numeric offset or length.");
    // Both are widened to uint64_t below, where a negative value would wrap to
    // a huge offset. The file-bounds check downstream would still reject it,
    // but only after reporting an absurd number, so reject it here instead.
    VELOX_CHECK_GE(
        offset->asInt(), 0, "Puffin blob metadata has a negative offset.");
    VELOX_CHECK_GE(
        length->asInt(), 0, "Puffin blob metadata has a negative length.");
    VELOX_CHECK(
        !found.has_value(),
        "Puffin file has multiple deletion vectors and no referenced data "
        "file to disambiguate them.");
    found = BlobLocation{
        static_cast<uint64_t>(offset->asInt()),
        static_cast<uint64_t>(length->asInt())};
  }

  VELOX_CHECK(
      found.has_value(),
      "Puffin footer has no deletion-vector blob for data file: {}",
      referencedDataFile.empty() ? std::string_view{"(unspecified)"}
                                 : std::string_view{referencedDataFile});
  return found.value();
}

// Unwraps the Iceberg deletion-vector-v1 blob frame
// ([length: 4B BE][magic][bitmap][CRC-32: 4B BE]), validating the magic and
// CRC-32, and returns a view of the inner roaring bitmap. If 'blob' is not
// framed (legacy raw roaring bitmap), returns it unchanged.
std::string_view unframeDeletionVector(std::string_view blob) {
  constexpr size_t kLengthSize = kDeletionVectorLengthSize;
  constexpr size_t kCrcSize = kDeletionVectorCrcSize;
  constexpr size_t kHeaderSize = kLengthSize + kDeletionVectorMagicSize;
  if (blob.size() < kHeaderSize ||
      std::memcmp(
          blob.data() + kLengthSize,
          kDeletionVectorMagic,
          kDeletionVectorMagicSize) != 0) {
    return blob;
  }

  const uint32_t magicAndVectorLength = readBigEndian32(blob.data());
  VELOX_CHECK_GE(
      magicAndVectorLength,
      kDeletionVectorMagicSize,
      "Deletion-vector-v1 length too small: {}.",
      magicAndVectorLength);
  VELOX_CHECK_EQ(
      blob.size(),
      kLengthSize + magicAndVectorLength + kCrcSize,
      "Deletion-vector-v1 blob size mismatch.");

  const uint32_t storedCrc =
      readBigEndian32(blob.data() + kLengthSize + magicAndVectorLength);
  // Iceberg stores the standard CRC-32 (java.util.zip.CRC32) over magic +
  // bitmap. zlib's crc32 is that same finalized IEEE 802.3 CRC-32.
  uLong crc = crc32(0L, Z_NULL, 0);
  crc = crc32(
      crc,
      reinterpret_cast<const Bytef*>(blob.data() + kLengthSize),
      static_cast<uInt>(magicAndVectorLength));
  const auto computedCrc = static_cast<uint32_t>(crc);
  VELOX_CHECK_EQ(storedCrc, computedCrc, "Deletion-vector-v1 CRC-32 mismatch.");

  return blob.substr(
      kHeaderSize, magicAndVectorLength - kDeletionVectorMagicSize);
}

// Detects whether 'bitmap' is a 64-bit or legacy 32-bit roaring bitmap and
// parses it.
dwio::common::RoaringBitmapView parseRoaringBitmap(std::string_view bitmap) {
  if (bitmap.size() < 8) {
    VELOX_FAIL(
        "Deletion vector blob too small: {} bytes, expected at least 8.",
        bitmap.size());
  }

  // Peek at first 4 bytes to detect 32-bit vs 64-bit format. A legacy 32-bit
  // RoaringBitmap holds positions in [0, 2^32).
  const uint32_t firstWord = readLittleEndian32(bitmap.data());
  const bool is32BitFormat = (firstWord == kSerialCookieNoRun) ||
      ((firstWord & 0xFFFF) == kSerialCookie);
  return dwio::common::RoaringBitmapView(
      bitmap,
      is32BitFormat ? dwio::common::RoaringBitmapView::Format::k32Bit
                    : dwio::common::RoaringBitmapView::Format::k64Bit);
}
} // namespace

static_assert(
    kMaxRoaring64GroupKey == dwio::common::RoaringBitmapView::kMaxGroupKey,
    "The reader and the writer must agree on the largest group key.");

DeletionVectorReader::DeletionVectorReader(
    const IcebergDeleteFile& dvFile,
    uint64_t splitOffset,
    memory::MemoryPool* pool,
    std::shared_ptr<const config::ConfigBase> connectorConfig,
    folly::F14FastMap<std::string, std::string> fileReadOps,
    std::shared_ptr<filesystems::TokenProvider> tokenProvider)
    : connector::DeletionVectorReader(
          dvFile.filePath,
          splitOffset,
          pool,
          std::move(connectorConfig),
          std::move(fileReadOps),
          std::move(tokenProvider)),
      dvFile_(dvFile) {
  VELOX_CHECK(
      dvFile_.content == FileContent::kDeletionVector,
      "Expected deletion vector file but got content type: {}",
      static_cast<int>(dvFile_.content));
  VELOX_CHECK_GT(dvFile_.recordCount, 0, "Empty deletion vector.");

  static constexpr int64_t kMaxDeletionVectorRecordCount = 10'000'000'000LL;
  VELOX_CHECK_LE(
      dvFile_.recordCount,
      kMaxDeletionVectorRecordCount,
      "Deletion vector record count exceeds sanity limit: {}",
      dvFile_.recordCount);
}

DeletionVectorReader::Bitmap DeletionVectorReader::readBitmap(
    ReadFile& file,
    const FileIoContext& ioContext) {
  // Prefer the typed contentOffset / contentLength fields. The legacy
  // bounds-map encoding (kDvOffsetFieldId / kDvLengthFieldId) is kept as a
  // fallback for callers that have not migrated yet.
  uint64_t blobOffset = static_cast<uint64_t>(dvFile_.contentOffset);
  uint64_t blobLength = dvFile_.contentLength > 0
      ? static_cast<uint64_t>(dvFile_.contentLength)
      : dvFile_.fileSizeInBytes;
  bool haveBlobLocation = dvFile_.contentLength > 0;

  if (dvFile_.contentLength == 0) {
    bool haveOffset = false;
    bool haveLength = false;
    if (auto it = dvFile_.lowerBounds.find(kDvOffsetFieldId);
        it != dvFile_.lowerBounds.end()) {
      try {
        blobOffset = std::stoull(it->second);
        haveOffset = true;
      } catch (const std::exception& e) {
        VELOX_FAIL(
            "Failed to parse DV blob offset from bounds map: {}", e.what());
      }
    }
    if (auto it = dvFile_.upperBounds.find(kDvLengthFieldId);
        it != dvFile_.upperBounds.end()) {
      try {
        blobLength = std::stoull(it->second);
        haveLength = true;
      } catch (const std::exception& e) {
        VELOX_FAIL(
            "Failed to parse DV blob length from bounds map: {}", e.what());
      }
    }
    // A legacy bounds-map location is usable only when both offset and length
    // are present. A partial entry falls through to Puffin-footer parsing
    // rather than unframing at a wrong offset/length.
    haveBlobLocation = haveOffset && haveLength;
  }

  const auto fileSize = file.size();

  // Nothing in the manifest located the blob. A Puffin file describes its own
  // blobs, so parse the footer rather than guessing. Non-Puffin inputs keep
  // the legacy whole-file behaviour, which is how raw roaring blobs are read.
  if (!haveBlobLocation && startsWithPuffinMagic(file, ioContext)) {
    const auto location =
        locateBlobFromPuffinFooter(file, dvFile_.referencedDataFile, ioContext);
    blobOffset = location.offset;
    blobLength = location.length;
  }

  VELOX_CHECK_LE(
      blobOffset,
      fileSize,
      "DV blob offset {} exceeds file size {}.",
      blobOffset,
      fileSize);
  VELOX_CHECK_LE(
      blobLength,
      fileSize - blobOffset,
      "DV blob range [{}, {}) exceeds file size {}.",
      blobOffset,
      blobOffset + blobLength,
      fileSize);

  auto blob = readBytes(file, blobOffset, blobLength, ioContext);

  // Unwrap the Iceberg deletion-vector-v1 frame (validates magic + CRC-32),
  // then parse the inner roaring bitmap. Legacy unframed blobs pass through.
  auto bitmap = parseRoaringBitmap(
      unframeDeletionVector(std::string_view(blob->as<char>(), blobLength)));
  return {std::move(blob), std::move(bitmap)};
}

const std::vector<int64_t>& DeletionVectorReader::deletedPositions() {
  const auto& deletionVector = bitmap();
  if (!positionsExpanded_) {
    positionsExpanded_ = true;
    deletedPositions_ = deletionVector.positions();
  }
  return deletedPositions_;
}

} // namespace facebook::velox::connector::hive::iceberg
