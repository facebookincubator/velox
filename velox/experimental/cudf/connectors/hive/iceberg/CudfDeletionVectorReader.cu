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

#include <cudf/column/column.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

using Roaring32BitmapType = cuco::experimental::
    roaring_bitmap<cuda::std::uint32_t, rmm::mr::polymorphic_allocator<char>>;
using Roaring64BitmapType = cuco::experimental::
    roaring_bitmap<cuda::std::uint64_t, rmm::mr::polymorphic_allocator<char>>;

struct NegateBool {
  __device__ bool operator()(bool b) const {
    return !b;
  }
};

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

/// Converts a run-encoded Roaring32 (cookie 12347, numContainers < 4) to
/// a no-run Roaring32 (cookie 12346) by expanding runs into sorted arrays.
/// The result always includes offset headers.
///
/// cuco's metadata parser rejects run bitmaps with < 4 containers outright
/// (the `contains_run_container` device code exists but the host parser
/// doesn't reach it), so we must convert to array format on the host.
std::string convertRunToNoRun(const char* r32, uint32_t numContainers) {
  std::size_t kcOffset = 4 + (numContainers + 7) / 8;

  // Parse key-card pairs and run data from the original block.
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

  // Build cookie-12346 format with offset headers.
  std::string out;
  uint32_t cookie = kNoRunCookie;
  out.append(reinterpret_cast<const char*>(&cookie), 4);
  out.append(reinterpret_cast<const char*>(&numContainers), 4);

  for (auto& ci : containers) {
    uint16_t cardMinus1 = static_cast<uint16_t>(ci.expandedValues.size() - 1);
    out.append(reinterpret_cast<const char*>(&ci.key), 2);
    out.append(reinterpret_cast<const char*>(&cardMinus1), 2);
  }

  // Offset section.
  uint32_t base = 8 + numContainers * 4 + numContainers * 4;
  for (auto& ci : containers) {
    out.append(reinterpret_cast<const char*>(&base), 4);
    base += static_cast<uint32_t>(ci.expandedValues.size()) * 2;
  }

  // Array data.
  for (auto& ci : containers) {
    for (auto v : ci.expandedValues) {
      out.append(reinterpret_cast<const char*>(&v), 2);
    }
  }

  return out;
}

/// Normalizes a single Roaring32 portable block for cuco.
/// - Cookie 12346, numContainers < 4: injects missing offset headers.
/// - Cookie 12347, numContainers < 4: converts runs to arrays (cookie 12346).
/// - Otherwise: returns original data unchanged.
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

/// Walks the Roaring64 portable payload and normalizes each Roaring32 bucket
/// for cuco. Returns the original data as-is if no bucket needs fixup.
std::string normalizeRoaring64ForCuco(const char* data, std::size_t size) {
  if (size < sizeof(uint64_t)) {
    return std::string(data, size);
  }

  uint64_t numBuckets;
  std::memcpy(&numBuckets, data, sizeof(uint64_t));

  // Fast path: check if any bucket needs fixup.
  bool needsFixup = false;
  std::size_t pos = sizeof(uint64_t);
  for (uint64_t b = 0; b < numBuckets && pos + 4 <= size; ++b) {
    pos += sizeof(uint32_t);
    if (pos + 4 > size) {
      break;
    }
    if (roaring32NeedsNormalization(data + pos, size - pos)) {
      needsFixup = true;
      break;
    }
    pos += roaring32BlockSize(data + pos, size - pos);
  }

  if (!needsFixup) {
    return std::string(data, size);
  }

  // Slow path: rebuild payload with normalized buckets.
  std::string out;
  out.reserve(size + numBuckets * 16);
  out.append(data, sizeof(uint64_t));
  pos = sizeof(uint64_t);

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

} // namespace

// ---------------------------------------------------------------------------
// BitmapImpl — opaque wrapper kept out of the header.
// Holds either a 32-bit or 64-bit cuco roaring bitmap depending on whether
// the input was raw Roaring32 or Roaring64 (inside a DV-v1 Puffin blob).
// ---------------------------------------------------------------------------

struct CudfDeletionVectorReader::BitmapImpl {
  std::unique_ptr<Roaring32BitmapType> bitmap32;
  std::unique_ptr<Roaring64BitmapType> bitmap64;

  template <class InputIt, class OutputIt>
  void contains(
      InputIt first,
      InputIt last,
      OutputIt out,
      rmm::cuda_stream_view stream) {
    if (bitmap64) {
      bitmap64->contains(first, last, out, stream);
    } else if (bitmap32) {
      bitmap32->contains(first, last, out, stream);
    }
  }

  bool empty() const {
    return !bitmap32 && !bitmap64;
  }
};

// ---------------------------------------------------------------------------
// Special members (all defined here where BitmapImpl is complete)
// ---------------------------------------------------------------------------

CudfDeletionVectorReader::CudfDeletionVectorReader(
    std::string filePath,
    uint64_t fileSizeInBytes,
    std::unordered_map<int32_t, std::string> lowerBounds,
    std::unordered_map<int32_t, std::string> upperBounds)
    : filePath_(std::move(filePath)),
      fileSizeInBytes_(fileSizeInBytes),
      lowerBounds_(std::move(lowerBounds)),
      upperBounds_(std::move(upperBounds)) {}

CudfDeletionVectorReader::~CudfDeletionVectorReader() = default;
CudfDeletionVectorReader::CudfDeletionVectorReader(
    CudfDeletionVectorReader&&) noexcept = default;
CudfDeletionVectorReader& CudfDeletionVectorReader::operator=(
    CudfDeletionVectorReader&&) noexcept = default;

// ---------------------------------------------------------------------------
// loadAndInitialize — load blob + parse envelope + build GPU bitmap
// ---------------------------------------------------------------------------

void CudfDeletionVectorReader::loadAndInitialize(rmm::cuda_stream_view stream) {
  dvBlobBytes_ = loadBlob();
  parseDvBlobEnvelope();

  const char* payload = dvBlobBytes_.data() + dvPayloadOffset_;
  std::size_t payloadSize = dvPayloadSize_;

  CUDF_EXPECTS(payloadSize >= 4, "Deletion vector payload too small");

  // Detect whether the payload is Roaring64 or raw Roaring32.
  // Roaring64 starts with a uint64 numBuckets, then per-bucket [uint32 key]
  // [Roaring32]. Raw Roaring32 starts with cookie 12346 or 12347.
  // Heuristic: if the first 4 bytes are a valid Roaring32 cookie, treat as
  // raw Roaring32. Otherwise treat as Roaring64.
  uint32_t firstWord;
  std::memcpy(&firstWord, payload, 4);
  bool isRawRoaring32 =
      (firstWord == kNoRunCookie) || ((firstWord & kCookieMask) == kRunCookie);

  bitmap_ = std::make_unique<BitmapImpl>();

  if (isRawRoaring32) {
    // Payload is already a raw Roaring32 block — use the 32-bit bitmap.
    normalizedPayload_ = normalizeRoaring32ForCuco(payload, payloadSize);
    auto const* bytes =
        reinterpret_cast<cuda::std::byte const*>(normalizedPayload_.data());
    bitmap_->bitmap32 = std::make_unique<Roaring32BitmapType>(
        bytes, rmm::mr::polymorphic_allocator<char>{}, stream);
  } else {
    // Payload is Roaring64: [uint64 numBuckets] per bucket: [uint32 key]
    // [Roaring32].
    CUDF_EXPECTS(
        payloadSize > sizeof(uint64_t),
        "Deletion vector Roaring64 payload too small");

    uint64_t numBuckets;
    std::memcpy(&numBuckets, payload, sizeof(uint64_t));

    if (numBuckets == 1) {
      // Single bucket — unwrap the inner Roaring32 and use the cheaper
      // 32-bit bitmap directly, avoiding the Roaring64 envelope overhead.
      const char* r32 = payload + sizeof(uint64_t) + sizeof(uint32_t);
      std::size_t r32Size = payloadSize - sizeof(uint64_t) - sizeof(uint32_t);
      normalizedPayload_ = normalizeRoaring32ForCuco(r32, r32Size);
      auto const* bytes =
          reinterpret_cast<cuda::std::byte const*>(normalizedPayload_.data());
      bitmap_->bitmap32 = std::make_unique<Roaring32BitmapType>(
          bytes, rmm::mr::polymorphic_allocator<char>{}, stream);
    } else {
      // Multiple buckets — must use 64-bit bitmap.
      normalizedPayload_ = normalizeRoaring64ForCuco(payload, payloadSize);
      auto const* bytes =
          reinterpret_cast<cuda::std::byte const*>(normalizedPayload_.data());
      bitmap_->bitmap64 = std::make_unique<Roaring64BitmapType>(
          bytes, rmm::mr::polymorphic_allocator<char>{}, stream);
    }
  }
}

// ---------------------------------------------------------------------------
// applyDeletionVector — filter deleted rows from a table chunk
// ---------------------------------------------------------------------------

std::unique_ptr<cudf::table> CudfDeletionVectorReader::applyDeletionVector(
    cudf::table_view const& table,
    std::size_t startRow,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const numRows = table.num_rows();
  if (numRows == 0 || !bitmap_ || bitmap_->empty()) {
    return std::make_unique<cudf::table>(table, stream, mr);
  }

  auto rowMask = rmm::device_buffer(numRows * sizeof(bool), stream, mr);
  auto rowMaskIter = thrust::make_transform_output_iterator(
      static_cast<bool*>(rowMask.data()), NegateBool{});

  if (bitmap_->bitmap32) {
    // 32-bit bitmap: row indices are uint32_t.
    auto rowIndices =
        rmm::device_buffer(numRows * sizeof(uint32_t), stream, mr);
    auto* rowIndicesPtr = static_cast<uint32_t*>(rowIndices.data());
    thrust::sequence(
        rmm::exec_policy_nosync(stream),
        rowIndicesPtr,
        rowIndicesPtr + numRows,
        static_cast<uint32_t>(startRow));
    bitmap_->bitmap32->contains(
        rowIndicesPtr, rowIndicesPtr + numRows, rowMaskIter, stream);
  } else {
    // 64-bit bitmap: row indices are std::size_t.
    auto rowIndices =
        rmm::device_buffer(numRows * sizeof(std::size_t), stream, mr);
    auto* rowIndicesPtr = static_cast<std::size_t*>(rowIndices.data());
    thrust::sequence(
        rmm::exec_policy_nosync(stream),
        rowIndicesPtr,
        rowIndicesPtr + numRows,
        startRow);
    bitmap_->bitmap64->contains(
        rowIndicesPtr, rowIndicesPtr + numRows, rowMaskIter, stream);
  }

  auto maskColumn = cudf::column_view(
      cudf::data_type{cudf::type_id::BOOL8},
      numRows,
      rowMask.data(),
      nullptr,
      0);

  return cudf::apply_boolean_mask(table, maskColumn, stream, mr);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
