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

#include <cstring>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

// ---------------------------------------------------------------------------
// DV-v1 envelope constants & helpers
// ---------------------------------------------------------------------------

static constexpr uint8_t kDvMagic[] = {0xD1, 0xD3, 0x39, 0x64};

constexpr auto readU32BigEndian(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
      (static_cast<uint32_t>(p[1]) << 16) | (static_cast<uint32_t>(p[2]) << 8) |
      static_cast<uint32_t>(p[3]);
}

// ---------------------------------------------------------------------------
// Roaring normalization helpers (pure CPU, moved from .cu)
// ---------------------------------------------------------------------------

constexpr uint32_t kNoRunCookie = 12346;
constexpr uint32_t kRunCookie = 12347;
constexpr uint32_t kCookieMask = 0xFFFF;
constexpr uint32_t kNoOffsetThreshold = 4;
constexpr uint32_t kMaxArrayContainerCard = 4096;
constexpr uint32_t kBitsetContainerBytes = 8192;

void parseRoaring32Cookie(
    const char* r32,
    uint32_t& cookie,
    uint32_t& numContainers) {
  std::memcpy(&cookie, r32, 4);
  if ((cookie & kCookieMask) == kRunCookie) {
    numContainers = (cookie >> 16) + 1;
  } else {
    std::memcpy(&numContainers, r32 + 4, 4);
  }
}

/// Returns true if the Roaring32 block at \p r32 needs normalization for cuco.
/// Two cases:
///   1. Cookie 12346 (no-run), numContainers in [1,3]: missing offset headers.
///   2. Cookie 12347 (run), numContainers < 4: cuco rejects this outright;
///      must convert to cookie 12346 (expand runs to arrays).
bool roaring32NeedsNormalization(const char* r32, std::size_t available) {
  if (available < 4) {
    return false;
  }
  uint32_t cookie, numContainers;
  parseRoaring32Cookie(r32, cookie, numContainers);
  bool isNoRun = (cookie == kNoRunCookie);
  bool isRun = ((cookie & kCookieMask) == kRunCookie);
  if (isNoRun) {
    return numContainers > 0 && numContainers < kNoOffsetThreshold;
  }
  if (isRun) {
    return numContainers < kNoOffsetThreshold;
  }
  return false;
}

/// For cookie 12346, returns the total serialized block size.
std::size_t
noRunBlockSize(const char* r32, uint32_t numContainers, bool hasOffsets) {
  std::size_t hdr = 8 + numContainers * 4;
  if (hasOffsets) {
    hdr += numContainers * 4;
  }
  std::size_t dataSize = 0;
  for (uint32_t c = 0; c < numContainers; ++c) {
    uint16_t cardMinus1;
    std::memcpy(&cardMinus1, r32 + 8 + c * 4 + 2, 2);
    uint32_t card = static_cast<uint32_t>(cardMinus1) + 1;
    dataSize +=
        (card <= kMaxArrayContainerCard) ? card * 2 : kBitsetContainerBytes;
  }
  return hdr + dataSize;
}

/// For cookie 12347, returns total serialized block size.
std::size_t
runBlockSize(const char* r32, std::size_t available, uint32_t numContainers) {
  std::size_t kcOffset = 4 + (numContainers + 7) / 8;
  bool hasOffsets = (numContainers >= kNoOffsetThreshold);
  std::size_t hdr = kcOffset + numContainers * 4;
  if (hasOffsets) {
    hdr += numContainers * 4;
  }
  const char* ptr = r32 + hdr;
  const char* end = r32 + available;
  for (uint32_t c = 0; c < numContainers && ptr + 2 <= end; ++c) {
    uint16_t numRuns;
    std::memcpy(&numRuns, ptr, 2);
    ptr += 2 + numRuns * 4;
  }
  return static_cast<std::size_t>(ptr - r32);
}

std::size_t roaring32BlockSize(const char* r32, std::size_t available) {
  if (available < 4) {
    return available;
  }
  uint32_t cookie, numContainers;
  parseRoaring32Cookie(r32, cookie, numContainers);
  if ((cookie & kCookieMask) == kRunCookie) {
    return runBlockSize(r32, available, numContainers);
  }
  return noRunBlockSize(
      r32, numContainers, numContainers >= kNoOffsetThreshold);
}

/// Injects missing offset headers into a cookie-12346 (no-run) Roaring32
/// bitmap with numContainers < 4.
std::string injectNoRunOffsets(
    const char* r32,
    std::size_t r32Size,
    uint32_t numContainers) {
  std::size_t headerEnd = 8 + numContainers * 4;
  std::string out;
  out.reserve(r32Size + numContainers * 4);
  out.append(r32, headerEnd);

  uint32_t base = static_cast<uint32_t>(headerEnd + numContainers * 4);
  for (uint32_t c = 0; c < numContainers; ++c) {
    out.append(reinterpret_cast<const char*>(&base), 4);
    uint16_t cardMinus1;
    std::memcpy(&cardMinus1, r32 + 8 + c * 4 + 2, 2);
    uint32_t card = static_cast<uint32_t>(cardMinus1) + 1;
    base += (card <= kMaxArrayContainerCard) ? card * 2 : kBitsetContainerBytes;
  }

  out.append(r32 + headerEnd, r32Size - headerEnd);
  return out;
}

/// Reads a cookie-12346 (no-run) Roaring32 bitmap with numContainers < 4
/// directly from file into \p out, injecting the missing offset headers
/// in-place. Avoids an intermediate temp buffer.
void injectNoRunOffsetsFromFile(
    velox::ReadFile& file,
    uint64_t fileOffset,
    std::size_t payloadSize,
    uint32_t numContainers,
    std::string& out) {
  std::size_t headerEnd = 8 + numContainers * 4;
  std::size_t offsetSectionSize = numContainers * 4;
  out.resize(payloadSize + offsetSectionSize);

  // Read the key-card header directly into the output.
  file.pread(fileOffset, headerEnd, out.data());

  // Compute and write offset entries in-place.
  char* offsetDst = out.data() + headerEnd;
  uint32_t base = static_cast<uint32_t>(headerEnd + offsetSectionSize);
  for (uint32_t c = 0; c < numContainers; ++c) {
    std::memcpy(offsetDst + c * 4, &base, 4);
    uint16_t cardMinus1;
    std::memcpy(&cardMinus1, out.data() + 8 + c * 4 + 2, 2);
    uint32_t card = static_cast<uint32_t>(cardMinus1) + 1;
    base += (card <= kMaxArrayContainerCard) ? card * 2 : kBitsetContainerBytes;
  }

  // Read the container data directly after the injected offsets.
  std::size_t dataSize = payloadSize - headerEnd;
  file.pread(
      fileOffset + headerEnd,
      dataSize,
      out.data() + headerEnd + offsetSectionSize);
}

/// Converts a run-encoded Roaring32 (cookie 12347, numContainers < 4) to
/// a no-run Roaring32 (cookie 12346) by expanding runs into sorted arrays.
/// The result always includes offset headers.
///
/// cuco's metadata parser rejects run bitmaps with < 4 containers outright
/// (the `contains_run_container` device code exists but the host parser
/// doesn't reach it), so we must convert to array format on the host.
std::string convertRunToNoRun(const char* r32, uint32_t numContainers) {
  std::size_t kcOffset = 4 + (numContainers + 7) / 8;

  struct ContainerInfo {
    uint16_t key;
    std::vector<uint16_t> expandedValues;
  };
  std::vector<ContainerInfo> containers(numContainers);

  const char* dataPtr = r32 + kcOffset + numContainers * 4;
  for (uint32_t c = 0; c < numContainers; ++c) {
    uint16_t key;
    std::memcpy(&key, r32 + kcOffset + c * 4, 2);
    containers[c].key = key;

    uint16_t numRuns;
    std::memcpy(&numRuns, dataPtr, 2);
    dataPtr += 2;
    for (uint16_t r = 0; r < numRuns; ++r) {
      uint16_t start, lenMinus1;
      std::memcpy(&start, dataPtr, 2);
      std::memcpy(&lenMinus1, dataPtr + 2, 2);
      dataPtr += 4;
      for (uint32_t v = start; v <= static_cast<uint32_t>(start) + lenMinus1;
           ++v) {
        containers[c].expandedValues.push_back(static_cast<uint16_t>(v));
      }
    }
  }

  std::string out;
  uint32_t cookie = kNoRunCookie;
  out.append(reinterpret_cast<const char*>(&cookie), 4);
  out.append(reinterpret_cast<const char*>(&numContainers), 4);

  for (auto& ci : containers) {
    uint16_t cardMinus1 = static_cast<uint16_t>(ci.expandedValues.size() - 1);
    out.append(reinterpret_cast<const char*>(&ci.key), 2);
    out.append(reinterpret_cast<const char*>(&cardMinus1), 2);
  }

  uint32_t base = 8 + numContainers * 4 + numContainers * 4;
  for (auto& ci : containers) {
    out.append(reinterpret_cast<const char*>(&base), 4);
    base += static_cast<uint32_t>(ci.expandedValues.size()) * 2;
  }

  for (auto& ci : containers) {
    for (auto v : ci.expandedValues) {
      out.append(reinterpret_cast<const char*>(&v), 2);
    }
  }

  return out;
}

/// Normalizes a single Roaring32 portable block for cuco.
std::string normalizeRoaring32ForCuco(const char* data, std::size_t size) {
  if (!roaring32NeedsNormalization(data, size)) {
    return std::string(data, size);
  }
  uint32_t cookie, numContainers;
  parseRoaring32Cookie(data, cookie, numContainers);
  if ((cookie & kCookieMask) == kRunCookie) {
    return convertRunToNoRun(data, numContainers);
  }
  std::size_t blockSize = noRunBlockSize(data, numContainers, false);
  return injectNoRunOffsets(data, blockSize, numContainers);
}

/// Returns true if any Roaring32 bucket inside a Roaring64 payload needs
/// normalization.
bool roaring64NeedsNormalization(const char* data, std::size_t size) {
  if (size < sizeof(uint64_t)) {
    return false;
  }
  uint64_t numBuckets;
  std::memcpy(&numBuckets, data, sizeof(uint64_t));
  std::size_t pos = sizeof(uint64_t);
  for (uint64_t b = 0; b < numBuckets && pos + 4 <= size; ++b) {
    pos += sizeof(uint32_t);
    if (pos + 4 > size) {
      break;
    }
    if (roaring32NeedsNormalization(data + pos, size - pos)) {
      return true;
    }
    pos += roaring32BlockSize(data + pos, size - pos);
  }
  return false;
}

/// Walks the Roaring64 portable payload and normalizes each Roaring32 bucket
/// for cuco.
std::string normalizeRoaring64ForCuco(const char* data, std::size_t size) {
  if (size < sizeof(uint64_t)) {
    return std::string(data, size);
  }

  uint64_t numBuckets;
  std::memcpy(&numBuckets, data, sizeof(uint64_t));

  std::string out;
  out.reserve(size + numBuckets * 16);
  out.append(data, sizeof(uint64_t));
  std::size_t pos = sizeof(uint64_t);

  for (uint64_t b = 0; b < numBuckets && pos + 4 <= size; ++b) {
    out.append(data + pos, sizeof(uint32_t));
    pos += sizeof(uint32_t);

    if (pos + 4 > size) {
      out.append(data + pos, size - pos);
      break;
    }

    std::size_t r32Total = roaring32BlockSize(data + pos, size - pos);

    if (roaring32NeedsNormalization(data + pos, size - pos)) {
      out.append(normalizeRoaring32ForCuco(data + pos, r32Total));
    } else {
      out.append(data + pos, r32Total);
    }
    pos += r32Total;
  }

  return out;
}

// ---------------------------------------------------------------------------
// Blob I/O helpers
// ---------------------------------------------------------------------------

struct BlobSource {
  std::shared_ptr<velox::ReadFile> file;
  uint64_t blobOffset{0};
  uint64_t blobLength{0};
  std::size_t payloadOffset{0};
  std::size_t payloadSize{0};
};

BlobSource openAndProbeBlob(
    const std::string& filePath,
    uint64_t fileSizeInBytes,
    const std::unordered_map<int32_t, std::string>& lowerBounds,
    const std::unordered_map<int32_t, std::string>& upperBounds) {
  BlobSource src;

  src.blobOffset = 0;
  src.blobLength = fileSizeInBytes;

  if (auto it = lowerBounds.find(CudfDeletionVectorReader::kDvOffsetFieldId);
      it != lowerBounds.end()) {
    try {
      src.blobOffset = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob offset from bounds map: {}", e.what());
    }
  }
  if (auto it = upperBounds.find(CudfDeletionVectorReader::kDvLengthFieldId);
      it != upperBounds.end()) {
    try {
      src.blobLength = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob length from bounds map: {}", e.what());
    }
  }

  auto fs = filesystems::getFileSystem(filePath, nullptr);
  src.file = fs->openFileForRead(filePath);

  auto fileSize = src.file->size();
  VELOX_CHECK_LE(
      src.blobOffset,
      fileSize,
      "DV blob offset {} exceeds file size {}.",
      src.blobOffset,
      fileSize);
  VELOX_CHECK_LE(
      src.blobLength,
      fileSize - src.blobOffset,
      "DV blob range [{}, {}) exceeds file size {}.",
      src.blobOffset,
      src.blobOffset + src.blobLength,
      fileSize);

  // Read just the DV-v1 envelope header (up to 12 bytes) to determine
  // payload offset and size without reading the entire blob.
  // DV-v1 format: [4B BE combined_length] [4B magic] [payload...] [4B CRC]
  constexpr std::size_t kEnvelopeHeaderSize = 12;
  if (src.blobLength >= kEnvelopeHeaderSize) {
    uint8_t hdr[kEnvelopeHeaderSize];
    src.file->pread(src.blobOffset, kEnvelopeHeaderSize, hdr);
    if (std::memcmp(hdr + 4, kDvMagic, 4) == 0) {
      const auto combinedLength = readU32BigEndian(hdr);
      if (combinedLength >= 4 &&
          src.blobLength >= static_cast<uint64_t>(4 + combinedLength + 4)) {
        src.payloadOffset = 8;
        src.payloadSize = combinedLength - 4;
        return src;
      }
    }
  }

  src.payloadOffset = 0;
  src.payloadSize = src.blobLength;
  return src;
}

} // namespace

// ---------------------------------------------------------------------------
// loadAndInitialize — probe header + single-copy payload read + GPU bitmap
// ---------------------------------------------------------------------------

void CudfDeletionVectorReader::loadAndInitialize(rmm::cuda_stream_view stream) {
  auto src =
      openAndProbeBlob(filePath_, fileSizeInBytes_, lowerBounds_, upperBounds_);
  const auto payloadFileOffset = src.blobOffset + src.payloadOffset;
  const auto payloadSize = src.payloadSize;

  VELOX_CHECK_GE(payloadSize, 4, "Deletion vector payload too small");

  // Read a small probe from the start of the payload to determine the
  // Roaring format and whether normalization is needed.  24 bytes covers:
  //   - Roaring32 cookie (4B) + numContainers (4B)
  //   - Roaring64 numBuckets (8B) + bucket key (4B) + inner cookie (4B) +
  //     inner numContainers (4B)
  constexpr std::size_t kProbeSize = 24;
  char probe[kProbeSize];
  auto probeBytes = std::min(payloadSize, kProbeSize);
  src.file->pread(payloadFileOffset, probeBytes, probe);

  uint32_t firstWord;
  std::memcpy(&firstWord, probe, sizeof(uint32_t));
  bool isRawRoaring32 =
      (firstWord == kNoRunCookie) || ((firstWord & kCookieMask) == kRunCookie);

  if (isRawRoaring32) {
    if (roaring32NeedsNormalization(probe, probeBytes)) {
      uint32_t cookie, numContainers;
      parseRoaring32Cookie(probe, cookie, numContainers);
      if (cookie == kNoRunCookie) {
        injectNoRunOffsetsFromFile(
            *src.file,
            payloadFileOffset,
            payloadSize,
            numContainers,
            normalizedPayload_);
      } else {
        // Run-cookie: must parse full payload to expand runs (tiny payloads).
        std::string tmp(payloadSize, '\0');
        src.file->pread(payloadFileOffset, payloadSize, tmp.data());
        normalizedPayload_ = convertRunToNoRun(tmp.data(), numContainers);
      }
    } else {
      normalizedPayload_.resize(payloadSize);
      src.file->pread(
          payloadFileOffset, payloadSize, normalizedPayload_.data());
    }
    buildBitmap<BitmapType::k32Bit>(stream);
  } else {
    VELOX_CHECK_GT(
        payloadSize,
        sizeof(uint64_t),
        "Deletion vector Roaring64 payload too small");

    uint64_t numBuckets;
    std::memcpy(&numBuckets, probe, sizeof(uint64_t));

    if (numBuckets == 1) {
      const auto r32FileOffset =
          payloadFileOffset + sizeof(uint64_t) + sizeof(uint32_t);
      const auto r32Size = payloadSize - sizeof(uint64_t) - sizeof(uint32_t);

      const char* innerProbe = probe + sizeof(uint64_t) + sizeof(uint32_t);
      auto innerProbeBytes = probeBytes - sizeof(uint64_t) - sizeof(uint32_t);

      if (roaring32NeedsNormalization(innerProbe, innerProbeBytes)) {
        uint32_t cookie, numContainers;
        parseRoaring32Cookie(innerProbe, cookie, numContainers);
        if (cookie == kNoRunCookie) {
          injectNoRunOffsetsFromFile(
              *src.file,
              r32FileOffset,
              r32Size,
              numContainers,
              normalizedPayload_);
        } else {
          std::string tmp(r32Size, '\0');
          src.file->pread(r32FileOffset, r32Size, tmp.data());
          normalizedPayload_ = convertRunToNoRun(tmp.data(), numContainers);
        }
      } else {
        normalizedPayload_.resize(r32Size);
        src.file->pread(r32FileOffset, r32Size, normalizedPayload_.data());
      }
      buildBitmap<BitmapType::k32Bit>(stream);
    } else {
      std::string tmp(payloadSize, '\0');
      src.file->pread(payloadFileOffset, payloadSize, tmp.data());
      if (roaring64NeedsNormalization(tmp.data(), tmp.size())) {
        normalizedPayload_ = normalizeRoaring64ForCuco(tmp.data(), tmp.size());
      } else {
        normalizedPayload_ = std::move(tmp);
      }
      buildBitmap<BitmapType::k64Bit>(stream);
    }
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
