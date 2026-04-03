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
#include "velox/common/file/FileSystems.h"

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

#include <cstring>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

using roaring_bitmap_type =
    cuco::experimental::roaring_bitmap<
        cuda::std::uint64_t,
        rmm::mr::polymorphic_allocator<char>>;

struct NegateBool {
  __device__ bool operator()(bool b) const {
    return !b;
  }
};

static constexpr uint8_t kDvMagic[] = {0xD1, 0xD3, 0x39, 0x64};

uint32_t readU32BE(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
      (static_cast<uint32_t>(p[1]) << 16) |
      (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

} // namespace

// ---------------------------------------------------------------------------
// BitmapImpl pimpl
// ---------------------------------------------------------------------------

struct CudfDeletionVectorReader::BitmapImpl {
  std::unique_ptr<roaring_bitmap_type> bitmap;
};

// ---------------------------------------------------------------------------
// CudfDeletionVectorReader
// ---------------------------------------------------------------------------

CudfDeletionVectorReader::CudfDeletionVectorReader(
    const IcebergDeleteFile& dvFile)
    : dvFile_(dvFile) {
  VELOX_CHECK(
      dvFile_.content ==
          ::facebook::velox::connector::hive::iceberg::FileContent::
              kDeletionVector,
      "Expected deletion vector file but got content type: {}",
      static_cast<int>(dvFile_.content));
}

CudfDeletionVectorReader::~CudfDeletionVectorReader() = default;
CudfDeletionVectorReader::CudfDeletionVectorReader(
    CudfDeletionVectorReader&&) noexcept = default;
CudfDeletionVectorReader& CudfDeletionVectorReader::operator=(
    CudfDeletionVectorReader&&) noexcept = default;

void CudfDeletionVectorReader::loadAndInitialize(
    rmm::cuda_stream_view stream) {
  dvBlobBytes_ = loadBlob();
  parseDvBlobEnvelope();

  VELOX_CHECK_GT(
      dvPayloadSize_,
      sizeof(uint64_t),
      "Deletion vector Roaring64 payload too small: {} bytes.",
      dvPayloadSize_);

  auto const* payloadBytes = reinterpret_cast<cuda::std::byte const*>(
      dvBlobBytes_.data() + dvPayloadOffset_);
  bitmap_ = std::make_unique<BitmapImpl>();
  bitmap_->bitmap = std::make_unique<roaring_bitmap_type>(
      payloadBytes, rmm::mr::polymorphic_allocator<char>{}, stream);
}

std::unique_ptr<cudf::table> CudfDeletionVectorReader::applyDeletionVector(
    cudf::table_view const& table,
    std::size_t startRow,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const numRows = table.num_rows();
  if (numRows == 0 || !bitmap_ || !bitmap_->bitmap) {
    return std::make_unique<cudf::table>(table, stream, mr);
  }

  // Build sequential row indices [startRow, startRow + numRows).
  auto rowIndices =
      rmm::device_buffer(numRows * sizeof(std::size_t), stream, mr);
  auto* rowIndicesPtr = static_cast<std::size_t*>(rowIndices.data());
  thrust::sequence(
      rmm::exec_policy_nosync(stream),
      rowIndicesPtr,
      rowIndicesPtr + numRows,
      startRow);

  // Query bitmap; negate output so true == "keep this row".
  auto rowMask = rmm::device_buffer(numRows * sizeof(bool), stream, mr);
  auto rowMaskIter = thrust::make_transform_output_iterator(
      static_cast<bool*>(rowMask.data()), NegateBool{});

  bitmap_->bitmap->contains(
      rowIndicesPtr, rowIndicesPtr + numRows, rowMaskIter, stream);

  auto maskColumn = cudf::column_view(
      cudf::data_type{cudf::type_id::BOOL8},
      numRows,
      rowMask.data(),
      nullptr,
      0);

  return cudf::apply_boolean_mask(table, maskColumn, stream, mr);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::string CudfDeletionVectorReader::loadBlob() {
  uint64_t blobOffset = 0;
  uint64_t blobLength = dvFile_.fileSizeInBytes;

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
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob length from bounds map: {}", e.what());
    }
  }

  auto fs = filesystems::getFileSystem(dvFile_.filePath, nullptr);
  auto readFile = fs->openFileForRead(dvFile_.filePath);

  auto fileSize = readFile->size();
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

  return blobData;
}

void CudfDeletionVectorReader::parseDvBlobEnvelope() {
  // DV-v1 blob format:
  //   [4B BE combined_length] [4B magic] [vector payload ...] [4B BE CRC]
  if (dvBlobBytes_.size() >= 12) {
    const auto* raw =
        reinterpret_cast<const uint8_t*>(dvBlobBytes_.data());
    if (std::memcmp(raw + 4, kDvMagic, 4) == 0) {
      uint32_t combinedLength = readU32BE(raw);
      if (combinedLength >= 4 &&
          dvBlobBytes_.size() >=
              static_cast<std::size_t>(4 + combinedLength + 4)) {
        dvPayloadOffset_ = 8;
        dvPayloadSize_ = combinedLength - 4;
        return;
      }
    }
  }

  // No wrapper detected; treat the entire blob as raw Roaring64 payload.
  dvPayloadOffset_ = 0;
  dvPayloadSize_ = dvBlobBytes_.size();
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
