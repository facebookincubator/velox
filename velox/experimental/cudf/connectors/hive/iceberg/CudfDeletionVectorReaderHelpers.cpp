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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReaderHelpers.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/FileSystems.h"

#include <cuda/iterator>

#include <algorithm>
#include <bit>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

/// Constants for the 32 bit roaring bitmap cookies.
constexpr uint32_t kNoRunCookie = 12346;
constexpr uint32_t kRunCookie = 12347;
constexpr uint32_t kCookieMask = 0xFFFF;
constexpr std::size_t kCookieSize = sizeof(uint32_t);

/// Constants for the 32 bit roaring bitmap.
constexpr uint32_t kNoOffsetThreshold = 4;
constexpr uint32_t kMaxArrayContainerCard = 4096;
constexpr uint32_t kBitsetContainerBytes = 8192;

constexpr std::size_t kContainerCountSize = sizeof(uint32_t);
constexpr std::size_t kNoRunHeaderPrefix = kCookieSize + kContainerCountSize;
constexpr std::size_t kKeyCardDescSize = sizeof(uint16_t) + sizeof(uint16_t);
constexpr std::size_t kOffsetEntrySize = sizeof(uint32_t);

constexpr std::size_t kRunPairSize = sizeof(uint16_t) + sizeof(uint16_t);
constexpr std::size_t kNumRunsSize = sizeof(uint16_t);

/// Reads an integral big endian value from a byte array.
template <typename T>
constexpr T readBigEndian(const uint8_t* p)
  requires(std::is_integral_v<T>)
{
  if constexpr (std::endian::native == std::endian::big) {
    return unalignedLoad<T>(
        std::string_view(reinterpret_cast<const char*>(p), sizeof(T)));
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

/// Checks if there's a valid 32 bit roaring bitmap cookie at the start of the
/// payload string.
bool isRoaring32Cookie(std::string_view payload) {
  auto cookie = unalignedLoad<uint32_t>(payload);
  return cookie == kNoRunCookie or ((cookie & kCookieMask) == kRunCookie);
}

/// Parses the first bytes of a 32-bit roaring bitmap in portable format to
/// extract the cookie and container count.
///   - No-run (cookie == 12346): followed by a 4-byte numContainers field.
///   - Run (cookie & 0xFFFF == 12347): upper 16 bits encode numContainers - 1.
std::pair<uint32_t, uint32_t> parseRoaring32Cookie(std::string_view payload) {
  VELOX_CHECK_GE(payload.size(), kCookieSize, "Roaring32 block too small");
  const auto cookie = unalignedLoad<uint32_t>(payload);

  if (cookie == kNoRunCookie) {
    const auto numContainers =
        unalignedLoad<uint32_t>(payload.substr(kCookieSize));
    return {cookie, numContainers};
  } else if ((cookie & kCookieMask) == kRunCookie) {
    return {cookie, (cookie >> 16) + 1};
  }

  VELOX_FAIL("Invalid 32-bit roaring bitmap cookie: {}", cookie);
}

/// Roaring32 portable layout (no-run variant):
///   [cookie          4B]  (uint32 == 12346)
///   [numContainers   4B]  (uint32)
///   [key-card descriptors  numContainers * 4B]  (key:u16, cardMinus1:u16)
///   [offset table    numContainers * 4B]  (only if numContainers >= 4)
///   [container data  variable]
///
/// Computes the total serialized block size for a block with no-run cookie
/// (12346) by walking the key-card descriptors to sum up container data sizes.
std::size_t noRunBlockSize(
    std::string_view payload,
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
        auto cardMinus1 = unalignedLoad<uint16_t>(
            payload,
            kNoRunHeaderPrefix + c * kKeyCardDescSize + sizeof(uint16_t));
        uint32_t card = static_cast<uint32_t>(cardMinus1) + 1;
        return acc +
            ((card <= kMaxArrayContainerCard) ? card * sizeof(uint16_t)
                                              : kBitsetContainerBytes);
      });
}

/// Roaring32 portable layout (run variant):
///   [cookie          4B]  (lower 16 bits == 12347, upper 16 = numContainers-1)
///   [run bitmap      ceil(numContainers/8) B]
///   [key-card descriptors  numContainers * 4B]
///   [offset table    numContainers * 4B]  (only if numContainers >= 4)
///   [run data        variable]  (numRuns:u16, then numRuns * (start:u16,
///   len-1:u16))
///
/// Computes the total serialized block size for a block with run cookie (12347)
/// by walking each container's run-length data.
std::size_t runBlockSize(std::string_view payload, uint32_t numContainers) {
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
        auto numRuns = unalignedLoad<uint16_t>(payload, acc);
        return acc + kNumRunsSize + numRuns * kRunPairSize;
      });
}

/// Computes the total serialized block size for a 32-bit roaring bitmap.
std::size_t roaring32BlockSize(std::string_view payload) {
  const auto [cookie, numContainers] = parseRoaring32Cookie(payload);
  if ((cookie & kCookieMask) == kRunCookie) {
    return runBlockSize(payload, numContainers);
  }
  return noRunBlockSize(
      payload, numContainers, numContainers >= kNoOffsetThreshold);
}

/// Injects the missing offset table into a no-run bitmap whose
/// numContainers < kNoOffsetThreshold. The offset table is placed between
/// the key-card descriptors and the container data.
std::string injectNoRunOffsets(
    std::string_view payload,
    uint32_t numContainers) {
  std::size_t headerEnd = kNoRunHeaderPrefix + numContainers * kKeyCardDescSize;
  std::size_t offsetSectionSize = numContainers * kOffsetEntrySize;
  std::string out;
  out.reserve(payload.size() + offsetSectionSize);
  out.append(payload.data(), headerEnd);

  // Compute cumulative offsets; base starts right after the injected offsets.
  uint32_t base = static_cast<uint32_t>(headerEnd + offsetSectionSize);
  std::for_each(
      cuda::counting_iterator<uint32_t>(0),
      cuda::counting_iterator<uint32_t>(numContainers),
      [&](auto container) {
        out.append(reinterpret_cast<const char*>(&base), kOffsetEntrySize);
        auto cardMinus1 = unalignedLoad<uint16_t>(
            payload,
            kNoRunHeaderPrefix + container * kKeyCardDescSize +
                sizeof(uint16_t));
        uint32_t card = static_cast<uint32_t>(cardMinus1) + 1;
        base += (card <= kMaxArrayContainerCard) ? card * sizeof(uint16_t)
                                                 : kBitsetContainerBytes;
      });

  out.append(payload.data() + headerEnd, payload.size() - headerEnd);
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
    std::string_view payload,
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
  std::for_each(
      cuda::counting_iterator<uint32_t>(0),
      cuda::counting_iterator<uint32_t>(numContainers),
      [&](auto container) {
        containers[container].key = unalignedLoad<uint16_t>(
            payload, kcOffset + container * kKeyCardDescSize);
        auto numRuns = unalignedLoad<uint16_t>(payload, dataPos);
        dataPos += kNumRunsSize;
        // For each run, extract the start and length and add the values to the
        // container
        std::for_each(
            cuda::counting_iterator<uint16_t>(0),
            cuda::counting_iterator<uint16_t>(numRuns),
            [&](auto) {
              auto start = static_cast<uint32_t>(
                  unalignedLoad<uint16_t>(payload, dataPos));
              auto length =
                  unalignedLoad<uint16_t>(payload, dataPos + sizeof(uint16_t)) +
                  1;
              dataPos += kRunPairSize;
              auto& vals = containers[container].expandedValues;
              std::transform(
                  cuda::counting_iterator(start),
                  cuda::counting_iterator(start + length),
                  std::back_inserter(vals),
                  [](auto value) { return static_cast<uint16_t>(value); });
            });
      });

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

} // namespace

bool is32bitBitmapNormalized(std::string_view payload) {
  // Get the cookie and the number of containers
  const auto [cookie, numContainers] = parseRoaring32Cookie(payload);
  if (cookie == kNoRunCookie) {
    return not(numContainers > 0 && numContainers < kNoOffsetThreshold);
  }
  return false;
}

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
        auto payload32 = payload.substr(pos);
        pos += roaring32BlockSize(payload32);
        return is32bitBitmapNormalized(payload32);
      });
}

std::string normalizeRoaring32(std::string_view payload) {
  const auto [cookie, numContainers] = parseRoaring32Cookie(payload);
  if ((cookie & kCookieMask) == kRunCookie) {
    return convertRunToNoRun(payload, numContainers);
  }
  std::size_t blockSize = noRunBlockSize(payload, numContainers, false);
  return injectNoRunOffsets(payload.substr(0, blockSize), numContainers);
}

std::string normalizeRoaring64(std::string_view payload, uint64_t numKeys) {
  std::string normalized;
  normalized.reserve(payload.size() + numKeys * 16);
  normalized.append(payload.data(), sizeof(uint64_t));
  std::size_t pos = sizeof(uint64_t);

  std::for_each(
      cuda::counting_iterator<uint64_t>(0),
      cuda::counting_iterator(numKeys),
      [&](auto) {
        // Skip over the key (4 bytes)
        if (pos + sizeof(uint32_t) > payload.size()) {
          return;
        }
        // Append the cookie
        normalized.append(payload.data() + pos, kCookieSize);
        pos += kCookieSize;

        // TODO(mh): Append the remaining data if less than 4 bytes are left.
        // Should we throw here instead?
        if (pos + sizeof(uint32_t) > payload.size()) {
          normalized.append(payload.data() + pos, payload.size() - pos);
          return;
        }

        auto payload32 = payload.substr(pos);
        const auto blockSize = roaring32BlockSize(payload32);
        // Append the 32 bit roaring bitmap
        if (is32bitBitmapNormalized(payload32)) {
          normalized.append(payload32.data(), blockSize);
        } else {
          normalized.append(normalizeRoaring32(payload32.substr(0, blockSize)));
        }
        pos += blockSize;
      });

  return normalized;
}

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

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
