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

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {
static constexpr uint32_t kSerialCookieNoRun = 12'346;
static constexpr uint32_t kSerialCookie = 12'347;
static constexpr uint32_t kRunContainersNoOffsetThreshold = 4;

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
bool startsWithPuffinMagic(ReadFile& file) {
  if (file.size() < kPuffinMagicSize) {
    return false;
  }
  std::string magic(kPuffinMagicSize, '\0');
  file.pread(0, kPuffinMagicSize, magic.data());
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
    const std::string& referencedDataFile) {
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
  file.pread(fileSize - kTrailerSize, kTrailerSize, trailer.data());
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
      payloadOffset - kPuffinMagicSize, kPuffinMagicSize, footerMagic.data());
  VELOX_CHECK_EQ(
      std::string_view(footerMagic),
      puffinMagic,
      "Puffin footer payload is not preceded by the expected magic.");

  std::string payload(payloadSize, '\0');
  file.pread(payloadOffset, payloadSize, payload.data());

  folly::dynamic footer;
  try {
    footer = folly::parseJson(payload);
  } catch (const folly::json::parse_error& e) {
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
std::string_view unframeDeletionVector(const std::string& blob) {
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

  return std::string_view(blob).substr(
      kHeaderSize, magicAndVectorLength - kDeletionVectorMagicSize);
}
} // namespace

DeletionVectorReader::DeletionVectorReader(
    const IcebergDeleteFile& dvFile,
    uint64_t splitOffset,
    memory::MemoryPool* /*pool*/,
    std::shared_ptr<const config::ConfigBase> connectorConfig)
    : dvFile_(dvFile),
      splitOffset_(splitOffset),
      connectorConfig_(std::move(connectorConfig)) {
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

void DeletionVectorReader::loadBitmap() {
  if (loaded_) {
    return;
  }
  loaded_ = true;

  // Prefer the typed contentOffset / contentLength fields. The legacy
  // bounds-map encoding (kDvOffsetFieldId / kDvLengthFieldId) is kept as a
  // fallback for callers that have not migrated yet.
  uint64_t blobOffset = static_cast<uint64_t>(dvFile_.contentOffset);
  uint64_t blobLength = dvFile_.contentLength > 0
      ? static_cast<uint64_t>(dvFile_.contentLength)
      : dvFile_.fileSizeInBytes;
  bool haveBlobLocation = dvFile_.contentLength > 0;

  if (dvFile_.contentLength == 0) {
    if (auto it = dvFile_.lowerBounds.find(kDvOffsetFieldId);
        it != dvFile_.lowerBounds.end()) {
      try {
        blobOffset = std::stoull(it->second);
      } catch (const std::exception& e) {
        VELOX_FAIL(
            "Failed to parse DV blob offset from bounds map: {}", e.what());
      }
    }
    if (auto it = dvFile_.upperBounds.find(kDvLengthFieldId);
        it != dvFile_.upperBounds.end()) {
      try {
        blobLength = std::stoull(it->second);
        haveBlobLocation = true;
      } catch (const std::exception& e) {
        VELOX_FAIL(
            "Failed to parse DV blob length from bounds map: {}", e.what());
      }
    }
  }

  // Pass the connector config (e.g. hive.manifold.* credentials) so
  // config-requiring filesystems like Manifold resolve a properly-configured
  // client for the Puffin read. Passing nullptr here previously left the
  // Manifold client unconfigured and segfaulted in openFileForRead.
  auto fs = filesystems::getFileSystem(dvFile_.filePath, connectorConfig_);
  VELOX_CHECK_NOT_NULL(
      fs,
      "No filesystem registered for deletion vector file: {}",
      dvFile_.filePath);
  auto readFile = fs->openFileForRead(dvFile_.filePath);
  VELOX_CHECK_NOT_NULL(
      readFile, "Failed to open deletion vector file: {}", dvFile_.filePath);

  auto fileSize = readFile->size();

  // Nothing in the manifest located the blob. A Puffin file describes its own
  // blobs, so parse the footer rather than guessing. Non-Puffin inputs keep
  // the legacy whole-file behaviour, which is how raw roaring blobs are read.
  if (!haveBlobLocation && startsWithPuffinMagic(*readFile)) {
    const auto location =
        locateBlobFromPuffinFooter(*readFile, dvFile_.referencedDataFile);
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

  std::string blobData(blobLength, '\0');
  readFile->pread(blobOffset, blobLength, blobData.data());

  // Unwrap the Iceberg deletion-vector-v1 frame (validates magic + CRC-32),
  // then parse the inner roaring bitmap. Legacy unframed blobs pass through.
  deserializeRoaring64Bitmap(unframeDeletionVector(blobData));

  std::sort(deletedPositions_.begin(), deletedPositions_.end());
}

void DeletionVectorReader::deserializeRoaring64Bitmap(std::string_view data) {
  if (data.size() < 8) {
    VELOX_FAIL(
        "Deletion vector blob too small: {} bytes, expected at least 8.",
        data.size());
  }

  const auto* ptr = reinterpret_cast<const uint8_t*>(data.data());
  const auto* end = ptr + data.size();

  // Peek at first 4 bytes to detect 32-bit vs 64-bit format.
  uint32_t firstWord;
  std::memcpy(&firstWord, ptr, sizeof(uint32_t));
  firstWord = folly::Endian::little(firstWord);

  bool is32BitFormat = (firstWord == kSerialCookieNoRun) ||
      ((firstWord & 0xFFFF) == kSerialCookie);

  if (is32BitFormat) {
    // Legacy 32-bit RoaringBitmap — all positions in [0, 2^32).
    deserialize32BitRoaringBitmap(ptr, end, 0);
    return;
  }

  // 64-bit Roaring64Bitmap format:
  //   [numGroups: uint64]
  //   For each group (sorted by highBits):
  //     [highBits: uint32]
  //     [32-bit RoaringBitmap in portable format]
  uint64_t numGroups;
  std::memcpy(&numGroups, ptr, sizeof(uint64_t));
  numGroups = folly::Endian::little(numGroups);
  ptr += sizeof(uint64_t);

  static constexpr uint64_t kMaxGroups = 1'000'000;
  VELOX_CHECK_LE(
      numGroups,
      kMaxGroups,
      "Roaring64Bitmap group count exceeds sanity limit: {}",
      numGroups);

  for (uint64_t g = 0; g < numGroups; ++g) {
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        sizeof(uint32_t),
        "Truncated Roaring64Bitmap group header.");

    uint32_t highBits;
    std::memcpy(&highBits, ptr, sizeof(uint32_t));
    highBits = folly::Endian::little(highBits);
    ptr += sizeof(uint32_t);

    VELOX_CHECK_LE(
        highBits,
        kMaxRoaring64GroupKey,
        "Roaring64Bitmap group key exceeds the maximum the Iceberg "
        "deletion-vector format can represent: {}",
        highBits);

    int64_t highBitsOffset = static_cast<int64_t>(highBits) << 32;

    // Deserialize the 32-bit bitmap for this group.
    // We need to find its size first by parsing the header.
    deserialize32BitRoaringBitmap(ptr, end, highBitsOffset);

    // Advance ptr past the 32-bit bitmap we just parsed.
    // Re-parse the header to compute the size.
    const auto* bitmapStart = ptr;

    uint32_t cookie;
    std::memcpy(&cookie, bitmapStart, sizeof(uint32_t));
    cookie = folly::Endian::little(cookie);

    bool hasRunContainers = false;
    uint32_t numContainers = 0;

    if ((cookie & 0xFFFF) == kSerialCookie) {
      hasRunContainers = true;
      numContainers = (cookie >> 16) + 1;
      ptr += sizeof(uint32_t);
    } else if (cookie == kSerialCookieNoRun) {
      ptr += sizeof(uint32_t);
      uint32_t containerCount;
      std::memcpy(&containerCount, ptr, sizeof(uint32_t));
      numContainers = folly::Endian::little(containerCount);
      ptr += sizeof(uint32_t);
    } else {
      VELOX_FAIL("Unknown roaring bitmap cookie in 64-bit group: {}", cookie);
    }

    if (numContainers == 0) {
      continue;
    }

    // Skip run bitmap if present.
    if (hasRunContainers) {
      uint32_t runBitmapBytes = (numContainers + 7) / 8;
      ptr += runBitmapBytes;
    }

    // Read key-cardinality pairs to compute container data sizes.
    struct ContainerMeta {
      uint16_t key;
      uint32_t cardinality;
      bool isRun;
    };
    std::vector<ContainerMeta> containers(numContainers);

    // Re-read run bitmap for container type detection.
    const auto* runBitmapPtr =
        hasRunContainers ? bitmapStart + sizeof(uint32_t) : nullptr;

    for (uint32_t i = 0; i < numContainers; ++i) {
      uint16_t key, cardMinus1;
      std::memcpy(&key, ptr, sizeof(uint16_t));
      key = folly::Endian::little(key);
      ptr += sizeof(uint16_t);
      std::memcpy(&cardMinus1, ptr, sizeof(uint16_t));
      cardMinus1 = folly::Endian::little(cardMinus1);
      ptr += sizeof(uint16_t);
      bool isRun = hasRunContainers && runBitmapPtr
          ? ((runBitmapPtr[i / 8] >> (i % 8)) & 1)
          : false;
      containers[i] = {key, static_cast<uint32_t>(cardMinus1) + 1, isRun};
    }

    // Skip offset section
    const bool hasOffsetSection =
        !hasRunContainers || numContainers >= kRunContainersNoOffsetThreshold;
    if (hasOffsetSection) {
      ptr += numContainers * sizeof(uint32_t);
    }

    // Skip container data.
    for (uint32_t i = 0; i < numContainers; ++i) {
      if (containers[i].isRun) {
        uint16_t numRuns;
        std::memcpy(&numRuns, ptr, sizeof(uint16_t));
        numRuns = folly::Endian::little(numRuns);
        ptr += sizeof(uint16_t) + static_cast<size_t>(numRuns) * 4;
      } else if (containers[i].cardinality <= 4'096) {
        ptr += static_cast<size_t>(containers[i].cardinality) * 2;
      } else {
        ptr += 8'192;
      }
    }
  }
}

void DeletionVectorReader::deserialize32BitRoaringBitmap(
    const uint8_t* ptr,
    const uint8_t* end,
    int64_t highBitsOffset) {
  VELOX_CHECK_GE(static_cast<size_t>(end - ptr), 8, "32-bit bitmap too small.");

  uint32_t cookie;
  std::memcpy(&cookie, ptr, sizeof(uint32_t));
  cookie = folly::Endian::little(cookie);
  ptr += sizeof(uint32_t);

  bool hasRunContainers = false;
  uint32_t numContainers = 0;

  if ((cookie & 0xFFFF) == kSerialCookie) {
    hasRunContainers = true;
    numContainers = (cookie >> 16) + 1;
  } else if (cookie == kSerialCookieNoRun) {
    std::memcpy(&numContainers, ptr, sizeof(uint32_t));
    numContainers = folly::Endian::little(numContainers);
    ptr += sizeof(uint32_t);
  } else {
    VELOX_FAIL(
        "Unknown roaring bitmap cookie: {}. Expected {} or {}.",
        cookie,
        kSerialCookieNoRun,
        kSerialCookie);
  }

  if (numContainers == 0) {
    return;
  }

  // Read run bitmap if present.
  std::vector<bool> isRunContainer(numContainers, false);
  if (hasRunContainers) {
    uint32_t runBitmapBytes = (numContainers + 7) / 8;
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        runBitmapBytes,
        "Truncated run bitmap.");
    for (uint32_t i = 0; i < numContainers; ++i) {
      isRunContainer[i] = (ptr[i / 8] >> (i % 8)) & 1;
    }
    ptr += runBitmapBytes;
  }

  // Read key-cardinality pairs.
  struct ContainerMeta {
    uint16_t key;
    uint32_t cardinality;
  };
  std::vector<ContainerMeta> containers(numContainers);

  VELOX_CHECK_GE(
      static_cast<size_t>(end - ptr),
      numContainers * 4,
      "Truncated container metadata.");
  for (uint32_t i = 0; i < numContainers; ++i) {
    uint16_t key, cardMinus1;
    std::memcpy(&key, ptr, sizeof(uint16_t));
    key = folly::Endian::little(key);
    ptr += sizeof(uint16_t);
    std::memcpy(&cardMinus1, ptr, sizeof(uint16_t));
    cardMinus1 = folly::Endian::little(cardMinus1);
    ptr += sizeof(uint16_t);
    containers[i] = {key, static_cast<uint32_t>(cardMinus1) + 1};
  }

  // Skip offset section
  const bool hasOffsetSection =
      !hasRunContainers || numContainers >= kRunContainersNoOffsetThreshold;
  if (hasOffsetSection) {
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        numContainers * 4,
        "Truncated offset section.");
    ptr += numContainers * sizeof(uint32_t);
  }

  // dvFile_.recordCount was already validated against
  // kMaxDeletionVectorRecordCount in the constructor.
  deletedPositions_.reserve(deletedPositions_.size() + dvFile_.recordCount);

  // Read container data.
  for (uint32_t i = 0; i < numContainers; ++i) {
    int64_t containerBase =
        highBitsOffset | (static_cast<int64_t>(containers[i].key) << 16);
    uint32_t cardinality = containers[i].cardinality;

    if (isRunContainer[i]) {
      uint16_t numRuns;
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr), 2u, "Truncated run container.");
      std::memcpy(&numRuns, ptr, sizeof(uint16_t));
      numRuns = folly::Endian::little(numRuns);
      ptr += sizeof(uint16_t);

      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          static_cast<size_t>(numRuns) * 4,
          "Truncated run container data.");
      for (uint16_t r = 0; r < numRuns; ++r) {
        uint16_t start, lengthMinus1;
        std::memcpy(&start, ptr, sizeof(uint16_t));
        start = folly::Endian::little(start);
        ptr += sizeof(uint16_t);
        std::memcpy(&lengthMinus1, ptr, sizeof(uint16_t));
        lengthMinus1 = folly::Endian::little(lengthMinus1);
        ptr += sizeof(uint16_t);
        for (uint32_t v = start;
             v <= static_cast<uint32_t>(start) + lengthMinus1;
             ++v) {
          deletedPositions_.push_back(containerBase | v);
        }
      }
    } else if (cardinality <= 4'096) {
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          cardinality * 2,
          "Truncated array container.");
      for (uint32_t j = 0; j < cardinality; ++j) {
        uint16_t val;
        std::memcpy(&val, ptr, sizeof(uint16_t));
        val = folly::Endian::little(val);
        ptr += sizeof(uint16_t);
        deletedPositions_.push_back(containerBase | val);
      }
    } else {
      static constexpr size_t kBitsetBytes = 8'192;
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          kBitsetBytes,
          "Truncated bitset container.");
      for (uint32_t word = 0; word < 1'024; ++word) {
        uint64_t bits;
        std::memcpy(&bits, ptr + word * 8, sizeof(uint64_t));
        bits = folly::Endian::little(bits);
        while (bits != 0) {
          uint32_t bit = __builtin_ctzll(bits);
          deletedPositions_.push_back(
              containerBase | static_cast<int64_t>(word * 64 + bit));
          bits &= bits - 1;
        }
      }
      ptr += kBitsetBytes;
    }
  }
}

void DeletionVectorReader::readDeletePositions(
    uint64_t baseReadOffset,
    uint64_t size,
    BufferPtr deleteBitmap) {
  loadBitmap();

  if (deletedPositions_.empty()) {
    return;
  }

  auto* bitmap = deleteBitmap->asMutable<uint8_t>();
  int64_t rowNumberLowerBound =
      static_cast<int64_t>(splitOffset_ + baseReadOffset);
  int64_t rowNumberUpperBound =
      rowNumberLowerBound + static_cast<int64_t>(size);

  while (positionIndex_ < deletedPositions_.size() &&
         deletedPositions_[positionIndex_] < rowNumberLowerBound) {
    ++positionIndex_;
  }

  while (positionIndex_ < deletedPositions_.size() &&
         deletedPositions_[positionIndex_] < rowNumberUpperBound) {
    auto bitIndex = static_cast<uint64_t>(
        deletedPositions_[positionIndex_] - rowNumberLowerBound);
    bits::setBit(bitmap, bitIndex);
    ++positionIndex_;
  }
}

bool DeletionVectorReader::noMoreData() const {
  return loaded_ && positionIndex_ >= deletedPositions_.size();
}

const std::vector<int64_t>& DeletionVectorReader::deletedPositions() {
  loadBitmap();
  return deletedPositions_;
}

} // namespace facebook::velox::connector::hive::iceberg
