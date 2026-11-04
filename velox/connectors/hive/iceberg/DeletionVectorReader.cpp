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

#include <algorithm>
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

uint16_t readLittleEndian16(const uint8_t* data) {
  uint16_t value;
  std::memcpy(&value, data, sizeof(value));
  return folly::Endian::little(value);
}

uint32_t readLittleEndian32(const uint8_t* data) {
  return readLittleEndian32(reinterpret_cast<const char*>(data));
}

uint64_t readLittleEndian64(const uint8_t* data) {
  uint64_t value;
  std::memcpy(&value, data, sizeof(value));
  return folly::Endian::little(value);
}

constexpr uint32_t kMaxContainers = 65'536;
constexpr uint32_t kMaxArrayContainerCardinality = 4'096;
constexpr size_t kBitsetContainerBytes = 8'192;
constexpr uint32_t kBitsetContainerWords = 1'024;
constexpr uint32_t kMaxContainerValue = 65'535;

// Sets bits [begin, end) of 'bitmap'. Writes only the bytes that hold those
// bits, so it never touches memory past a buffer sized for the batch.
void setBitRange(uint8_t* bitmap, uint64_t begin, uint64_t end) {
  while (begin < end && begin % 8 != 0) {
    bits::setBit(bitmap, begin++);
  }
  const uint64_t fullBytesEnd = end & ~uint64_t{7};
  if (begin < fullBytesEnd) {
    std::memset(bitmap + begin / 8, 0xFF, (fullBytesEnd - begin) / 8);
    begin = fullBytesEnd;
  }
  while (begin < end) {
    bits::setBit(bitmap, begin++);
  }
}

// The apply functions below set the bits of the container's positions that
// fall in [from, to). 'lower' is the row position of bit 0 in 'bitmap'. The
// caller guarantees base <= from < to <= base + 65536.

void applyArrayContainer(
    const uint8_t* values,
    uint32_t count,
    int64_t base,
    int64_t from,
    int64_t to,
    int64_t lower,
    uint8_t* bitmap) {
  const auto low = static_cast<uint32_t>(from - base);
  const auto high = static_cast<uint32_t>(to - base);
  uint32_t first = 0;
  uint32_t last = count;
  while (first < last) {
    const uint32_t mid = first + (last - first) / 2;
    if (readLittleEndian16(values + 2 * mid) < low) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  for (uint32_t i = first; i < count; ++i) {
    const uint32_t value = readLittleEndian16(values + 2 * i);
    if (value >= high) {
      break;
    }
    bits::setBit(bitmap, static_cast<uint64_t>(base + value - lower));
  }
}

void applyRunContainer(
    const uint8_t* runs,
    uint32_t numRuns,
    int64_t base,
    int64_t from,
    int64_t to,
    int64_t lower,
    uint8_t* bitmap) {
  const auto low = static_cast<uint32_t>(from - base);
  const auto high = static_cast<uint32_t>(to - base);
  // Runs are sorted and disjoint, so their last values increase as well.
  uint32_t first = 0;
  uint32_t last = numRuns;
  while (first < last) {
    const uint32_t mid = first + (last - first) / 2;
    const uint32_t runLast =
        static_cast<uint32_t>(readLittleEndian16(runs + 4 * mid)) +
        readLittleEndian16(runs + 4 * mid + 2);
    if (runLast < low) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  for (uint32_t i = first; i < numRuns; ++i) {
    const uint32_t start = readLittleEndian16(runs + 4 * i);
    if (start >= high) {
      break;
    }
    const uint32_t end = start + readLittleEndian16(runs + 4 * i + 2) + 1;
    setBitRange(
        bitmap,
        static_cast<uint64_t>(base + std::max(start, low) - lower),
        static_cast<uint64_t>(base + std::min(end, high) - lower));
  }
}

void applyBitsetContainer(
    const uint8_t* words,
    int64_t base,
    int64_t from,
    int64_t to,
    int64_t lower,
    uint8_t* bitmap) {
  const auto low = static_cast<uint32_t>(from - base);
  const auto high = static_cast<uint32_t>(to - base);
  const uint32_t firstWord = low / 64;
  const uint32_t lastWord = (high - 1) / 64;
  for (uint32_t w = firstWord; w <= lastWord; ++w) {
    uint64_t word = readLittleEndian64(words + 8 * w);
    if (w == firstWord) {
      word &= ~uint64_t{0} << (low % 64);
    }
    if (w == lastWord && high % 64 != 0) {
      word &= (uint64_t{1} << (high % 64)) - 1;
    }
    const int64_t wordBase = base + static_cast<int64_t>(w) * 64 - lower;
    while (word != 0) {
      bits::setBit(
          bitmap, static_cast<uint64_t>(wordBase + __builtin_ctzll(word)));
      word &= word - 1;
    }
  }
}

void appendArrayPositions(
    const uint8_t* values,
    uint32_t count,
    int64_t base,
    std::vector<int64_t>& positions) {
  for (uint32_t i = 0; i < count; ++i) {
    positions.push_back(base + readLittleEndian16(values + 2 * i));
  }
}

void appendRunPositions(
    const uint8_t* runs,
    uint32_t numRuns,
    int64_t base,
    std::vector<int64_t>& positions) {
  for (uint32_t i = 0; i < numRuns; ++i) {
    const uint32_t start = readLittleEndian16(runs + 4 * i);
    const uint32_t last = start + readLittleEndian16(runs + 4 * i + 2);
    for (uint32_t value = start; value <= last; ++value) {
      positions.push_back(base + value);
    }
  }
}

void appendBitsetPositions(
    const uint8_t* words,
    int64_t base,
    std::vector<int64_t>& positions) {
  for (uint32_t w = 0; w < kBitsetContainerWords; ++w) {
    uint64_t word = readLittleEndian64(words + 8 * w);
    while (word != 0) {
      positions.push_back(
          base + static_cast<int64_t>(w) * 64 + __builtin_ctzll(word));
      word &= word - 1;
    }
  }
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

  blob_.assign(blobLength, '\0');
  readFile->pread(blobOffset, blobLength, blob_.data());

  // Unwrap the Iceberg deletion-vector-v1 frame (validates magic + CRC-32),
  // then parse the inner roaring bitmap. Legacy unframed blobs pass through.
  const auto bitmap = unframeDeletionVector(blob_);
  bitmapOffset_ = static_cast<size_t>(bitmap.data() - blob_.data());
  bitmapSize_ = bitmap.size();
  parseRoaringBitmap();

  // Writers emit containers in increasing order. Sort anyway so an out-of-order
  // blob is still applied correctly; there are few containers per file.
  const auto byBase = [](const Container& lhs, const Container& rhs) {
    return lhs.base < rhs.base;
  };
  if (!std::is_sorted(containers_.begin(), containers_.end(), byBase)) {
    std::sort(containers_.begin(), containers_.end(), byBase);
  }
}

void DeletionVectorReader::parseRoaringBitmap() {
  if (bitmapSize_ < 8) {
    VELOX_FAIL(
        "Deletion vector blob too small: {} bytes, expected at least 8.",
        bitmapSize_);
  }
  const auto* data = bitmapData();

  // Peek at first 4 bytes to detect 32-bit vs 64-bit format.
  const uint32_t firstWord = readLittleEndian32(data);
  const bool is32BitFormat = (firstWord == kSerialCookieNoRun) ||
      ((firstWord & 0xFFFF) == kSerialCookie);
  if (is32BitFormat) {
    // Legacy 32-bit RoaringBitmap — all positions in [0, 2^32).
    parse32BitRoaringBitmap(0, 0);
    return;
  }

  // 64-bit Roaring64Bitmap format:
  //   [numGroups: uint64]
  //   For each group (sorted by highBits):
  //     [highBits: uint32]
  //     [32-bit RoaringBitmap in portable format]
  const uint64_t numGroups = readLittleEndian64(data);
  static constexpr uint64_t kMaxGroups = 1'000'000;
  VELOX_CHECK_LE(
      numGroups,
      kMaxGroups,
      "Roaring64Bitmap group count exceeds sanity limit: {}",
      numGroups);

  size_t offset = sizeof(uint64_t);
  for (uint64_t group = 0; group < numGroups; ++group) {
    VELOX_CHECK_GE(
        bitmapSize_ - offset,
        sizeof(uint32_t),
        "Truncated Roaring64Bitmap group header.");
    const uint32_t highBits = readLittleEndian32(data + offset);
    offset += sizeof(uint32_t);
    VELOX_CHECK_LE(
        highBits,
        kMaxRoaring64GroupKey,
        "Roaring64Bitmap group key exceeds the maximum the Iceberg "
        "deletion-vector format can represent: {}",
        highBits);
    offset =
        parse32BitRoaringBitmap(offset, static_cast<int64_t>(highBits) << 32);
  }
}

size_t DeletionVectorReader::parse32BitRoaringBitmap(
    size_t offset,
    int64_t highBitsOffset) {
  const auto* data = bitmapData();
  const auto remaining = [&]() { return bitmapSize_ - offset; };
  VELOX_CHECK_GE(remaining(), 8, "32-bit bitmap too small.");

  const uint32_t cookie = readLittleEndian32(data + offset);
  offset += sizeof(uint32_t);

  bool hasRunContainers = false;
  uint32_t numContainers = 0;
  if ((cookie & 0xFFFF) == kSerialCookie) {
    hasRunContainers = true;
    numContainers = (cookie >> 16) + 1;
  } else if (cookie == kSerialCookieNoRun) {
    numContainers = readLittleEndian32(data + offset);
    offset += sizeof(uint32_t);
  } else {
    VELOX_FAIL(
        "Unknown roaring bitmap cookie: {}. Expected {} or {}.",
        cookie,
        kSerialCookieNoRun,
        kSerialCookie);
  }

  if (numContainers == 0) {
    return offset;
  }
  VELOX_CHECK_LE(
      numContainers,
      kMaxContainers,
      "Roaring bitmap container count exceeds the 16-bit key space: {}",
      numContainers);

  const size_t runBitmapOffset = offset;
  if (hasRunContainers) {
    const size_t runBitmapBytes = (numContainers + 7) / 8;
    VELOX_CHECK_GE(remaining(), runBitmapBytes, "Truncated run bitmap.");
    offset += runBitmapBytes;
  }

  const size_t keysOffset = offset;
  VELOX_CHECK_GE(
      remaining(),
      static_cast<size_t>(numContainers) * 4,
      "Truncated container metadata.");
  offset += static_cast<size_t>(numContainers) * 4;

  const bool hasOffsetSection =
      !hasRunContainers || numContainers >= kRunContainersNoOffsetThreshold;
  if (hasOffsetSection) {
    VELOX_CHECK_GE(
        remaining(),
        static_cast<size_t>(numContainers) * 4,
        "Truncated offset section.");
    offset += static_cast<size_t>(numContainers) * sizeof(uint32_t);
  }

  containers_.reserve(containers_.size() + numContainers);
  for (uint32_t i = 0; i < numContainers; ++i) {
    const uint16_t key = readLittleEndian16(data + keysOffset + 4 * i);
    const uint32_t cardinality = static_cast<uint32_t>(readLittleEndian16(
                                     data + keysOffset + 4 * i + 2)) +
        1;
    const bool isRun =
        hasRunContainers && ((data[runBitmapOffset + i / 8] >> (i % 8)) & 1);
    const int64_t base = highBitsOffset | (static_cast<int64_t>(key) << 16);

    if (isRun) {
      VELOX_CHECK_GE(remaining(), 2, "Truncated run container.");
      const uint16_t numRuns = readLittleEndian16(data + offset);
      offset += sizeof(uint16_t);
      VELOX_CHECK_GE(
          remaining(),
          static_cast<size_t>(numRuns) * 4,
          "Truncated run container data.");
      // Applying a batch binary-searches the runs, so they must be sorted,
      // disjoint and inside the container. A violation would otherwise drop
      // deletes silently.
      int64_t previousLast = -1;
      for (uint16_t r = 0; r < numRuns; ++r) {
        const uint32_t start = readLittleEndian16(data + offset + 4 * r);
        const uint32_t last =
            start + readLittleEndian16(data + offset + 4 * r + 2);
        VELOX_CHECK_LE(
            last,
            kMaxContainerValue,
            "Roaring run extends past its container.");
        VELOX_CHECK_GT(
            static_cast<int64_t>(start),
            previousLast,
            "Roaring runs are not sorted and disjoint.");
        previousLast = last;
      }
      containers_.push_back(
          {base, base + previousLast, offset, numRuns, ContainerKind::kRun});
      offset += static_cast<size_t>(numRuns) * 4;
    } else if (cardinality <= kMaxArrayContainerCardinality) {
      VELOX_CHECK_GE(
          remaining(),
          static_cast<size_t>(cardinality) * 2,
          "Truncated array container.");
      // Applying a batch binary-searches the values, so they must be strictly
      // increasing. A violation would otherwise drop deletes silently.
      int32_t previous = -1;
      for (uint32_t j = 0; j < cardinality; ++j) {
        const int32_t value = readLittleEndian16(data + offset + 2 * j);
        VELOX_CHECK_GT(
            value,
            previous,
            "Roaring array container values are not strictly increasing.");
        previous = value;
      }
      containers_.push_back(
          {base, base + previous, offset, cardinality, ContainerKind::kArray});
      offset += static_cast<size_t>(cardinality) * 2;
    } else {
      VELOX_CHECK_GE(
          remaining(), kBitsetContainerBytes, "Truncated bitset container.");
      int64_t lastValue = -1;
      for (int32_t w = kBitsetContainerWords - 1; w >= 0; --w) {
        const uint64_t word = readLittleEndian64(data + offset + 8 * w);
        if (word != 0) {
          lastValue = static_cast<int64_t>(w) * 64 + 63 - __builtin_clzll(word);
          break;
        }
      }
      containers_.push_back(
          {base, base + lastValue, offset, 0, ContainerKind::kBitset});
      offset += kBitsetContainerBytes;
    }
  }
  return offset;
}

void DeletionVectorReader::readDeletePositions(
    uint64_t baseReadOffset,
    uint64_t size,
    BufferPtr deleteBitmap) {
  loadBitmap();

  const auto lower = static_cast<int64_t>(splitOffset_ + baseReadOffset);
  const auto upper = lower + static_cast<int64_t>(size);
  while (containerIndex_ < containers_.size() &&
         containers_[containerIndex_].lastPosition < lower) {
    ++containerIndex_;
  }

  const auto* data = bitmapData();
  for (size_t i = containerIndex_;
       i < containers_.size() && containers_[i].base < upper;
       ++i) {
    const auto& container = containers_[i];
    const int64_t from = std::max(lower, container.base);
    const int64_t to = std::min(upper, container.lastPosition + 1);
    if (from >= to) {
      continue;
    }
    auto* bitmap = deleteBitmap->asMutable<uint8_t>();
    const auto* payload = data + container.dataOffset;
    switch (container.kind) {
      case ContainerKind::kArray:
        applyArrayContainer(
            payload, container.count, container.base, from, to, lower, bitmap);
        break;
      case ContainerKind::kRun:
        applyRunContainer(
            payload, container.count, container.base, from, to, lower, bitmap);
        break;
      case ContainerKind::kBitset:
        applyBitsetContainer(payload, container.base, from, to, lower, bitmap);
        break;
    }
  }

  // A container whose last position is behind this batch is fully consumed.
  while (containerIndex_ < containers_.size() &&
         containers_[containerIndex_].lastPosition < upper) {
    ++containerIndex_;
  }
}

bool DeletionVectorReader::noMoreData() const {
  return loaded_ && containerIndex_ >= containers_.size();
}

const std::vector<int64_t>& DeletionVectorReader::deletedPositions() {
  loadBitmap();
  if (positionsExpanded_) {
    return deletedPositions_;
  }
  positionsExpanded_ = true;
  const auto* data = bitmapData();
  for (const auto& container : containers_) {
    const auto* payload = data + container.dataOffset;
    switch (container.kind) {
      case ContainerKind::kArray:
        appendArrayPositions(
            payload, container.count, container.base, deletedPositions_);
        break;
      case ContainerKind::kRun:
        appendRunPositions(
            payload, container.count, container.base, deletedPositions_);
        break;
      case ContainerKind::kBitset:
        appendBitsetPositions(payload, container.base, deletedPositions_);
        break;
    }
  }
  // Containers can share a key only in an out-of-spec blob.
  if (!std::is_sorted(deletedPositions_.begin(), deletedPositions_.end())) {
    std::sort(deletedPositions_.begin(), deletedPositions_.end());
  }
  return deletedPositions_;
}

} // namespace facebook::velox::connector::hive::iceberg
