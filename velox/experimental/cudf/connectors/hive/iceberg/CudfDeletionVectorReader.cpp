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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"

#include <cuda/iterator>

#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

namespace {

/// Reads an integral big endian value from a byte array.
template <typename T>
constexpr T readBigEndian(const uint8_t* p)
  requires(std::is_integral_v<T>)
{
  if constexpr (std::endian::native == std::endian::big) {
    T val;
    std::memcpy(&val, p, sizeof(val));
    return val;
  } else {
    // TODO(mh): Replace with std::byteswap(std::bit_cast<T>(p)) when C++23 is
    // available.
    auto val = T{0};
    // Fold to unroll shift left and OR ops at compile time
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((val = static_cast<T>(val << 8) | static_cast<T>(p[I])), ...);
    }(std::make_index_sequence<sizeof(T)>{});
    return val;
  }
}
} // namespace

namespace {

// Roaring32 portable layout (no-run variant):
//   [cookie          4B]  (uint32 == 12346)
//   [numContainers   4B]  (uint32)
//   [key-card descriptors  numContainers * 4B]  (key:u16, cardMinus1:u16)
//   [offset table    numContainers * 4B]  (only if numContainers >= 4)
//   [container data  variable]
//
// Roaring32 portable layout (run variant):
//   [cookie          4B]  (lower 16 bits == 12347, upper 16 = numContainers-1)
//   [run bitmap      ceil(numContainers/8) B]
//   [key-card descriptors  numContainers * 4B]
//   [offset table    numContainers * 4B]  (only if numContainers >= 4)
//   [run data        variable]  (numRuns:u16, then numRuns * (start:u16,
//   len-1:u16))

constexpr uint32_t kNoRunCookie = 12346;
constexpr uint32_t kRunCookie = 12347;
constexpr uint32_t kCookieMask = 0xFFFF;
constexpr uint32_t kNoOffsetThreshold = 4;
constexpr uint32_t kMaxArrayContainerCard = 4096;
constexpr uint32_t kBitsetContainerBytes = 8192;

constexpr std::size_t kCookieSize = sizeof(uint32_t);
constexpr std::size_t kContainerCountSize = sizeof(uint32_t);
constexpr std::size_t kNoRunHeaderPrefix = kCookieSize + kContainerCountSize;
constexpr std::size_t kKeyCardDescSize = sizeof(uint16_t) + sizeof(uint16_t);
constexpr std::size_t kOffsetEntrySize = sizeof(uint32_t);
constexpr std::size_t kRunPairSize = sizeof(uint16_t) + sizeof(uint16_t);
constexpr std::size_t kNumRunsSize = sizeof(uint16_t);

bool isRoaring32Cookie(std::string_view roaring32) {
  uint32_t cookie{0};
  std::memcpy(&cookie, roaring32.data(), sizeof(uint32_t));
  return cookie == kNoRunCookie or ((cookie & kCookieMask) == kRunCookie);
}

/// Parses the first bytes of a 32-bit roaring bitmap in portable format to
/// extract the cookie and container count.
///
/// Cookie variants:
///   - No-run (cookie == 12346): followed by a 4-byte numContainers field.
///   - Run (cookie & 0xFFFF == 12347): upper 16 bits encode numContainers - 1.
///
/// @param roaring32 String view over the serialized roaring32 data (>= 4
/// bytes).
/// @return A pair of {cookie, numContainers}.
std::pair<uint32_t, uint32_t> parseRoaring32Cookie(std::string_view roaring32) {
  VELOX_CHECK_GE(roaring32.size(), kCookieSize, "Roaring32 block too small");

  uint32_t cookie{0};
  std::memcpy(&cookie, roaring32.data(), sizeof(uint32_t));
  if (cookie == kNoRunCookie) {
    uint32_t numContainers{0};
    std::memcpy(
        &numContainers, roaring32.data() + sizeof(uint32_t), sizeof(uint32_t));
    return {cookie, numContainers};
  }
  if ((cookie & kCookieMask) == kRunCookie) {
    return {cookie, (cookie >> 16) + 1};
  }
  VELOX_FAIL("Invalid roaring32 cookie: {}", cookie);
}

/// Computes the total serialized block size for a block with no-run cookie
/// (12346) by walking the key-card descriptors to sum up container data sizes.
std::size_t noRunBlockSize(
    std::string_view roaring32,
    uint32_t numContainers,
    bool hasOffsets) {
  std::size_t hdr = kNoRunHeaderPrefix + numContainers * kKeyCardDescSize;
  if (hasOffsets) {
    hdr += numContainers * kOffsetEntrySize;
  }
  return std::accumulate(
      cuda::counting_iterator<uint32_t>(0),
      cuda::counting_iterator(numContainers),
      std::size_t{hdr},
      [&](std::size_t acc, uint32_t c) {
        uint16_t cardMinus1;
        std::memcpy(
            &cardMinus1,
            roaring32.data() + kNoRunHeaderPrefix + c * kKeyCardDescSize +
                sizeof(uint16_t),
            sizeof(uint16_t));
        uint32_t card = static_cast<uint32_t>(cardMinus1) + 1;
        return acc +
            ((card <= kMaxArrayContainerCard) ? card * sizeof(uint16_t)
                                              : kBitsetContainerBytes);
      });
}

/// Computes the total serialized block size for a block with run cookie (12347)
/// by walking each container's run-length data.
std::size_t runBlockSize(std::string_view roaring32, uint32_t numContainers) {
  // Run bitmap follows the 4-byte cookie, one bit per container.
  std::size_t runBitmapSize = (numContainers + 7) / 8;
  std::size_t kcOffset = kCookieSize + runBitmapSize;
  bool hasOffsets = (numContainers >= kNoOffsetThreshold);
  std::size_t hdr = kcOffset + numContainers * kKeyCardDescSize;
  if (hasOffsets) {
    hdr += numContainers * kOffsetEntrySize;
  }
  return std::accumulate(
      cuda::counting_iterator<uint32_t>(0),
      cuda::counting_iterator(numContainers),
      std::size_t{hdr},
      [&](std::size_t acc, uint32_t c) {
        uint16_t numRuns;
        std::memcpy(&numRuns, roaring32.data() + acc, kNumRunsSize);
        return acc + kNumRunsSize + numRuns * kRunPairSize;
      });
}

/// Computes the total serialized block size for a 32-bit roaring bitmap.
std::size_t roaring32BlockSize(std::string_view roaring32) {
  const auto [cookie, numContainers] = parseRoaring32Cookie(roaring32);
  if ((cookie & kCookieMask) == kRunCookie) {
    return runBlockSize(roaring32, numContainers);
  }
  return noRunBlockSize(
      roaring32, numContainers, numContainers >= kNoOffsetThreshold);
}

/// Checks whether the 32-bit roaring block is normalized for cuco.
///   - No-run cookie (12346), numContainers in [1,3]: offset table is omitted
///     per the portable spec but cuco requires it; must inject dummy offsets.
///   - Run cookie (12347): cuco only accepts the no-run portable format;
///     must convert run-encoded containers to array/bitset containers.
bool is32bitBitmapNormalized(std::string_view roaring32) {
  // Get the cookie and the number of containers
  const auto [cookie, numContainers] = parseRoaring32Cookie(roaring32);
  if (cookie == kNoRunCookie) {
    return not(numContainers > 0 && numContainers < kNoOffsetThreshold);
  }
  return false;
}

/// Checks if all 32 bit roaring bitmaps in the payload are normalized.
bool is64bitBitmapNormalized(std::string_view payload, uint64_t numKeys) {
  // Skip over the numKeys (8 bytes) prefix.
  std::size_t pos = sizeof(uint64_t);

  VELOX_CHECK_LE(
      pos + sizeof(uint32_t),
      payload.size(),
      "64-bit roaring payload is too small");

  return std::all_of(
      cuda::counting_iterator<uint64_t>(0),
      cuda::counting_iterator<uint64_t>(numKeys),
      [&](uint64_t key) {
        if (pos + sizeof(uint32_t) > payload.size()) {
          return true;
        }
        // Skip over the key (4 bytes)
        pos += sizeof(uint32_t);
        // Get the 32 bit roaring bitmap
        auto roaring32view = payload.substr(pos);
        pos += roaring32BlockSize(roaring32view);
        return is32bitBitmapNormalized(roaring32view);
      });
}

/// Injects the missing offset table into a no-run bitmap whose
/// numContainers < kNoOffsetThreshold. The offset table is placed between
/// the key-card descriptors and the container data.
std::string injectNoRunOffsets(
    std::string_view roaring32,
    uint32_t numContainers) {
  std::size_t headerEnd = kNoRunHeaderPrefix + numContainers * kKeyCardDescSize;
  std::size_t offsetSectionSize = numContainers * kOffsetEntrySize;
  std::string out;
  out.reserve(roaring32.size() + offsetSectionSize);
  out.append(roaring32.data(), headerEnd);

  // Compute cumulative offsets; base starts right after the injected offsets.
  uint32_t base = static_cast<uint32_t>(headerEnd + offsetSectionSize);
  for (uint32_t c = 0; c < numContainers; ++c) {
    out.append(reinterpret_cast<const char*>(&base), kOffsetEntrySize);
    uint16_t cardMinus1;
    std::memcpy(
        &cardMinus1,
        roaring32.data() + kNoRunHeaderPrefix + c * kKeyCardDescSize +
            sizeof(uint16_t),
        sizeof(uint16_t));
    uint32_t card = static_cast<uint32_t>(cardMinus1) + 1;
    base += (card <= kMaxArrayContainerCard) ? card * sizeof(uint16_t)
                                             : kBitsetContainerBytes;
  }

  out.append(roaring32.data() + headerEnd, roaring32.size() - headerEnd);
  return out;
}

/// Converts a run-encoded Roaring32 (cookie 12347, numContainers < 4) to
/// a no-run Roaring32 (cookie 12346) by expanding runs into sorted arrays.
/// The result always includes offset headers.
///
/// cuco's metadata parser rejects run bitmaps with < 4 containers outright
/// (the `contains_run_container` device code exists but the host parser
/// doesn't reach it), so we must convert to array format on the host.
std::string convertRunToNoRun(
    std::string_view roaring32,
    uint32_t numContainers) {
  // In the run variant the key-card descriptors start after the cookie and
  // the per-container run bitmap.
  std::size_t runBitmapSize = (numContainers + 7) / 8;
  std::size_t kcOffset = kCookieSize + runBitmapSize;

  struct ContainerInfo {
    uint16_t key;
    std::vector<uint16_t> expandedValues;
  };
  std::vector<ContainerInfo> containers(numContainers);

  // Run data follows the key-card descriptors (and offsets, if present).
  std::size_t dataPos = kcOffset + numContainers * kKeyCardDescSize;
  for (uint32_t c = 0; c < numContainers; ++c) {
    uint16_t key;
    std::memcpy(
        &key,
        roaring32.data() + kcOffset + c * kKeyCardDescSize,
        sizeof(uint16_t));
    containers[c].key = key;

    uint16_t numRuns;
    std::memcpy(&numRuns, roaring32.data() + dataPos, kNumRunsSize);
    dataPos += kNumRunsSize;
    for (uint16_t rr = 0; rr < numRuns; ++rr) {
      uint16_t start, lenMinus1;
      std::memcpy(&start, roaring32.data() + dataPos, sizeof(uint16_t));
      std::memcpy(
          &lenMinus1,
          roaring32.data() + dataPos + sizeof(uint16_t),
          sizeof(uint16_t));
      dataPos += kRunPairSize;
      for (uint32_t v = start; v <= static_cast<uint32_t>(start) + lenMinus1;
           ++v) {
        containers[c].expandedValues.push_back(static_cast<uint16_t>(v));
      }
    }
  }

  // Emit a no-run portable block with offsets always included.
  std::string out;
  uint32_t cookie = kNoRunCookie;
  out.append(reinterpret_cast<const char*>(&cookie), kCookieSize);
  out.append(
      reinterpret_cast<const char*>(&numContainers), kContainerCountSize);

  for (auto& ci : containers) {
    uint16_t cardMinus1 = static_cast<uint16_t>(ci.expandedValues.size() - 1);
    out.append(reinterpret_cast<const char*>(&ci.key), sizeof(uint16_t));
    out.append(reinterpret_cast<const char*>(&cardMinus1), sizeof(uint16_t));
  }

  // Offset table: each entry points to the start of that container's data.
  uint32_t base = static_cast<uint32_t>(
      kNoRunHeaderPrefix + numContainers * kKeyCardDescSize +
      numContainers * kOffsetEntrySize);
  for (auto& ci : containers) {
    out.append(reinterpret_cast<const char*>(&base), kOffsetEntrySize);
    base += static_cast<uint32_t>(ci.expandedValues.size()) * sizeof(uint16_t);
  }

  for (auto& ci : containers) {
    for (auto v : ci.expandedValues) {
      out.append(reinterpret_cast<const char*>(&v), sizeof(uint16_t));
    }
  }

  return out;
}

/// Normalizes a single 32 bit roaring bitmap for cuco.
std::string normalizeRoaring32(std::string_view data) {
  const auto [cookie, numContainers] = parseRoaring32Cookie(data);
  if ((cookie & kCookieMask) == kRunCookie) {
    return convertRunToNoRun(data, numContainers);
  }
  std::size_t blockSize = noRunBlockSize(data, numContainers, false);
  return injectNoRunOffsets(data.substr(0, blockSize), numContainers);
}

/// Walks the Roaring64 portable payload and normalizes each Roaring32 bucket
/// for cuco.
std::string normalizeRoaring64(std::string_view data) {
  VELOX_CHECK_GE(data.size(), sizeof(uint64_t), "Roaring64 payload too small");

  uint64_t numKeys;
  std::memcpy(&numKeys, data.data(), sizeof(uint64_t));

  std::string normalized;
  normalized.reserve(data.size() + numKeys * 16);
  normalized.append(data.data(), sizeof(uint64_t));
  std::size_t pos = sizeof(uint64_t);

  for (uint64_t b = 0; b < numKeys && pos + 4 <= data.size(); ++b) {
    normalized.append(data.data() + pos, kCookieSize);
    pos += kCookieSize;

    if (pos + 4 > data.size()) {
      normalized.append(data.data() + pos, data.size() - pos);
      break;
    }

    auto roaring32view = data.substr(pos);
    std::size_t roaring32Total = roaring32BlockSize(roaring32view);

    if (is32bitBitmapNormalized(roaring32view)) {
      normalized.append(roaring32view.data(), roaring32Total);
    } else {
      normalized.append(
          normalizeRoaring32(roaring32view.substr(0, roaring32Total)));
    }
    pos += roaring32Total;
  }

  return normalized;
}

/// Representation of deletion vector v1 (DV-v1) blob source.
struct BlobSource {
  std::shared_ptr<velox::ReadFile> file;
  std::size_t payloadFileOffset{0};
  std::size_t payloadSize{0};
  bool isRawRoaring32{false};
};

/// Loads a DV v1 blob from the file.
BlobSource loadBlobSource(
    const std::string_view filePath,
    uint64_t fileSizeInBytes,
    const std::unordered_map<int32_t, std::string>& lowerBounds,
    const std::unordered_map<int32_t, std::string>& upperBounds) {
  // Start with a raw DV blob source
  BlobSource source;

  uint64_t blobOffset = 0;
  uint64_t blobLength = fileSizeInBytes;

  // Read the envoded DB blob offset and length from the `IcebergDeleteFile`
  // bounds maps (encoded by the coordinator).
  if (auto it = lowerBounds.find(CudfDeletionVectorReader::kDvOffsetFieldId);
      it != lowerBounds.end()) {
    try {
      blobOffset = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob offset from bounds map: {}", e.what());
    }
  }
  if (auto it = upperBounds.find(CudfDeletionVectorReader::kDvLengthFieldId);
      it != upperBounds.end()) {
    try {
      blobLength = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob length from bounds map: {}", e.what());
    }
  }

  // Open the puffin file
  auto fs = filesystems::getFileSystem(filePath, nullptr);
  source.file = fs->openFileForRead(filePath);
  auto fileSize = source.file->size();

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

  // DV-v1 blob spec:
  //   [combined length of magic + payload (4 bytes - Big Endian)]
  //   [magic (4 bytes)]
  //   [payload (N bytes)]
  //   [CRC (4 bytes)]
  constexpr std::size_t kMagicOffset = sizeof(uint32_t);
  constexpr std::size_t kMagicSize = sizeof(uint32_t);
  constexpr std::size_t kCrcSize = sizeof(uint32_t);
  constexpr std::size_t kCombinedLengthSize = sizeof(uint32_t);

  // Envelope header size: combined length + magic + CRC sizes
  constexpr std::size_t kEnvelopeHeaderSize =
      kCombinedLengthSize + kMagicSize + kCrcSize;
  // Buffer to read at most the envelope header
  uint8_t hdr[kEnvelopeHeaderSize];
  const auto probeSize = std::min(blobLength, kEnvelopeHeaderSize);
  source.file->pread(blobOffset, probeSize, hdr);

  // Check if the probed bytes indicate a raw roaring32 bitmap
  VELOX_CHECK_GE(
      blobLength,
      kCookieSize,
      "DV blob too small: {} < {}",
      blobLength,
      kCookieSize);
  if (isRoaring32Cookie(
          std::string_view(reinterpret_cast<const char*>(hdr), kCookieSize))) {
    // Raw Roaring32 bitmap — the entire blob is the payload.
    source.payloadFileOffset = blobOffset;
    source.payloadSize = blobLength;
    source.isRawRoaring32 = true;
    return source;
  }

  // Ensure we have at least the envelope header size
  VELOX_CHECK_GE(
      blobLength,
      kEnvelopeHeaderSize,
      "DV blob too small: {} < {}",
      blobLength,
      kEnvelopeHeaderSize);

  const auto combinedLength = readBigEndian<uint32_t>(hdr);

  // Check the combined length and magic
  VELOX_CHECK_GE(
      combinedLength,
      kMagicSize,
      "DV-v1 combined length too small: {}.",
      combinedLength);

  constexpr uint8_t kDvMagic[kMagicSize] = {0xD1, 0xD3, 0x39, 0x64};
  VELOX_CHECK(
      std::memcmp(hdr + kMagicOffset, kDvMagic, kMagicSize) == 0,
      "DV-v1 magic mismatch in Puffin blob.");

  // Expected blob size: combined length + magic + CRC sizes
  const auto blobSize =
      static_cast<uint64_t>(kCombinedLengthSize + combinedLength + kCrcSize);
  VELOX_CHECK_EQ(blobLength, blobSize, "DV-v1 blob size mismatch.");

  // Payload file offset: blob start + combined length field + magic
  source.payloadFileOffset = blobOffset + kCombinedLengthSize + kMagicSize;
  // Payload size: combined length minus magic size
  source.payloadSize = combinedLength - kMagicSize;

  return source;
}

} // namespace

CudfDeletionVectorReader::CudfDeletionVectorReader(
    const velox_iceberg::IcebergDeleteFile& dvFile,
    uint64_t splitOffset)
    : dvFile_{dvFile.filePath, dvFile.fileSizeInBytes, dvFile.lowerBounds, dvFile.upperBounds},
      splitOffset_(splitOffset) {
  VELOX_CHECK(
      dvFile.content == velox_iceberg::FileContent::kDeletionVector,
      "Expected deletion vector file but got content type: {}",
      static_cast<int>(dvFile.content));
  VELOX_CHECK_GT(dvFile.recordCount, 0, "Empty deletion vector.");

  static constexpr int64_t kMaxDeletionVectorRecordCount = 10'000'000'000LL;
  VELOX_CHECK_LE(
      dvFile.recordCount,
      kMaxDeletionVectorRecordCount,
      "Empty deletion vector.");
}

void CudfDeletionVectorReader::loadBitmap(rmm::cuda_stream_view stream) {
  if (loaded_) {
    return;
  }

  // Puffin file layout:
  //   [Magic "PUF1" (4 bytes)]
  //   [Blob0 (M bytes)]
  //   [Blob1 (N bytes)]
  //   [Footer]
  //   [Footer payload size (4 bytes, little-endian)]
  //   [Flags (4 bytes)]
  //   [Magic "PUF1" (4 bytes)]
  const auto source = loadBlobSource(
      dvFile_.filePath,
      dvFile_.fileSizeInBytes,
      dvFile_.lowerBounds,
      dvFile_.upperBounds);

  // Read the payload from the file.
  auto payload = std::string{};
  payload.resize(source.payloadSize);
  source.file->pread(
      source.payloadFileOffset, source.payloadSize, payload.data());

  // Check if the payload is a raw roaring32 bitmap instead of a DV-v1 blob and
  // read it directly
  if (source.isRawRoaring32) {
    if (is32bitBitmapNormalized(payload)) {
      buildBitmap<BitmapType::k32Bit>(payload, stream);
    } else {
      auto normalizedPayload = normalizeRoaring32(payload);
      buildBitmap<BitmapType::k32Bit>(normalizedPayload, stream);
    }
    return;
  }

  // DV-v1 payload spec:
  //    [Number of keys (8 bytes)]
  //    [1st Key (4 bytes)]
  //    [32 bit roaring bitmap 1]
  //    ...
  //    [Nth Key (4 bytes)]
  //    [Nth 32 bit roaring bitmap]
  uint64_t numKeys;
  std::memcpy(&numKeys, payload.data(), sizeof(uint64_t));
  VELOX_CHECK_GT(numKeys, 0, "Deletion vector has zero keys");

  // Single key. Use 32-bit dispatch
  if (numKeys == 1) {
    // Skip the numKeys (8 bytes) + first key (4 bytes) prefix to get the
    // roaring32 bitmap.
    constexpr std::size_t kRoaring32Offset =
        sizeof(uint64_t) + sizeof(uint32_t);
    auto roaring32 = std::string_view(payload).substr(kRoaring32Offset);
    if (is32bitBitmapNormalized(roaring32)) {
      buildBitmap<BitmapType::k32Bit>(roaring32, stream);
    } else {
      auto normalizedPayload = normalizeRoaring32(roaring32);
      buildBitmap<BitmapType::k32Bit>(normalizedPayload, stream);
    }
  } else {
    // Multiple keys. Use 64-bit dispatch
    if (is64bitBitmapNormalized(payload, numKeys)) {
      buildBitmap<BitmapType::k64Bit>(payload, stream);
    } else {
      auto normalizedPayload = normalizeRoaring64(payload);
      buildBitmap<BitmapType::k64Bit>(normalizedPayload, stream);
    }
  }

  return;
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
