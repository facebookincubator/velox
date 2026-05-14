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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDeletionHelpers.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"

#include <cudf/column/column_factories.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;
namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

/// Roaring portable-format cookie constants.
constexpr uint32_t kNoRunCookie = 12346;
constexpr uint32_t kRunCookie = 12347;
constexpr uint32_t kCookieMask = 0xFFFF;
constexpr std::size_t kCookieSize = sizeof(uint32_t);

/// Loads a fixed width value from a string view without assuming aligned
/// memory.
template <typename T>
T inline unalignedLoad(std::string_view payload, std::size_t offset = 0)
  requires(std::is_integral_v<T>)
{
  T value;
  std::memcpy(&value, payload.data() + offset, sizeof(T));
  return value;
}

/// Reads an integral big endian value from a byte array.
template <typename T>
constexpr T inline readBigEndian(const uint8_t* p)
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
[[nodiscard]] bool inline isRoaring32Cookie(std::string_view payload) {
  auto cookie = unalignedLoad<uint32_t>(payload);
  return cookie == kNoRunCookie or ((cookie & kCookieMask) == kRunCookie);
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
  const auto source = loadBlobSource();

  // Read the payload from the file.
  auto payload = std::string{};
  payload.resize(source.payloadSize);
  source.file->pread(
      source.payloadFileOffset, source.payloadSize, payload.data());

  // Check if the payload is a raw roaring32 bitmap instead of a DV-v1 blob and
  // read it directly
  if (source.isRawRoaring32) {
    buildBitmap(cudf::roaring_bitmap_type::BITS_32, payload, stream);
    return;
  }

  // DV-v1 payload spec:
  //    [Number of keys (8 bytes)]
  //    [1st Key (4 bytes)]
  //    [32 bit roaring bitmap 1]
  //    ...
  //    [Nth Key (4 bytes)]
  //    [Nth 32 bit roaring bitmap]
  auto numKeys = unalignedLoad<uint64_t>(payload);
  VELOX_CHECK_GT(numKeys, 0, "Deletion vector has zero keys");

  // Single key. Use 32-bit dispatch
  if (numKeys == 1) {
    // Skip the numKeys (8 bytes) + first key (4 bytes) prefix to get the
    // roaring32 bitmap.
    constexpr std::size_t kRoaring32Offset =
        sizeof(uint64_t) + sizeof(uint32_t);
    auto roaring32 = std::string_view(payload).substr(kRoaring32Offset);
    buildBitmap(cudf::roaring_bitmap_type::BITS_32, roaring32, stream);
  } else {
    // Multiple keys. Use 64-bit dispatch
    buildBitmap(cudf::roaring_bitmap_type::BITS_64, payload, stream);
  }

  return;
}

CudfDeletionVectorReader::BlobSource
CudfDeletionVectorReader::loadBlobSource() {
  // Start with a raw DV blob source
  BlobSource source;

  uint64_t blobOffset = 0;
  uint64_t blobLength = dvFile_.fileSizeInBytes;

  // Read the envoded DB blob offset and length from the `IcebergDeleteFile`
  // bounds maps (encoded by the coordinator).
  if (auto it =
          dvFile_.lowerBounds.find(CudfDeletionVectorReader::kDvOffsetFieldId);
      it != dvFile_.lowerBounds.end()) {
    try {
      blobOffset = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob offset from bounds map: {}", e.what());
    }
  }
  if (auto it =
          dvFile_.upperBounds.find(CudfDeletionVectorReader::kDvLengthFieldId);
      it != dvFile_.upperBounds.end()) {
    try {
      blobLength = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob length from bounds map: {}", e.what());
    }
  }

  // Open the puffin file
  auto fs = filesystems::getFileSystem(dvFile_.filePath, nullptr);
  source.file = fs->openFileForRead(dvFile_.filePath);
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

/// Constructs the cuco roaring bitmap on the GPU from the roaringBitmapPayload.
void CudfDeletionVectorReader::buildBitmap(
    cudf::roaring_bitmap_type bitmapType,
    std::string_view roaringBitmapPayload,
    rmm::cuda_stream_view stream) {
  auto const* bitmapBytes =
      reinterpret_cast<cuda::std::byte const*>(roaringBitmapPayload.data());
  auto const bitmapSize = roaringBitmapPayload.size();

  bitmap_ = std::make_unique<cudf::roaring_bitmap>(
      bitmapType,
      cudf::host_span<const cuda::std::byte>(bitmapBytes, bitmapSize));

  // Load and mark the roaring bitmap as loaded
  bitmap_->materialize(stream);
  loaded_ = true;
}

void CudfDeletionVectorReader::applyDeletes(
    cudf::mutable_column_view const& deleteMask,
    std::size_t startRow,
    std::size_t numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr) {
  if (numRows == 0) {
    return;
  }

  // Load the cuco roaring bitmap
  loadBitmap(stream);

  // Return early if no bitmap or empty bitmap
  if (not bitmap_ or bitmap_->empty()) {
    return;
  }

  // Pick the row-index value type based on the bitmap key width.
  const auto valueTypeId =
      (bitmap_->type() == cudf::roaring_bitmap_type::BITS_32)
      ? cudf::type_to_id<uint32_t>()
      : cudf::type_to_id<uint64_t>();

  // Construct row index column if needed
  if (not rowIndices_ or rowIndices_->size() < numRows or
      rowIndices_->type().id() != valueTypeId) {
    rowIndices_ = cudf::make_numeric_column(
        cudf::data_type{valueTypeId},
        static_cast<cudf::size_type>(numRows),
        cudf::mask_state::UNALLOCATED,
        stream,
        temp_mr);
  }

  // Generate row indices
  if (bitmap_->type() == cudf::roaring_bitmap_type::BITS_32) {
    fillSequence<uint32_t>(
        rowIndices_->mutable_view(),
        static_cast<uint32_t>(startRow),
        static_cast<int64_t>(numRows),
        stream,
        temp_mr);
  } else {
    fillSequence<uint64_t>(
        rowIndices_->mutable_view(),
        static_cast<uint64_t>(startRow),
        static_cast<int64_t>(numRows),
        stream,
        temp_mr);
  }

  bitmap_->contains_async(rowIndices_->view(), deleteMask, stream);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
