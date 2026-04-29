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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDeletionHelpers.h"

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"

#include <cudf/column/column_factories.hpp>

#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

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
      buildBitmap(cudf::roaring_bitmap_type::BITS_32, payload, stream);
    } else {
      auto normalizedPayload = normalizeRoaring32(payload);
      buildBitmap(
          cudf::roaring_bitmap_type::BITS_32, normalizedPayload, stream);
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
  auto numKeys = unalignedLoad<uint64_t>(payload);
  VELOX_CHECK_GT(numKeys, 0, "Deletion vector has zero keys");

  // Single key. Use 32-bit dispatch
  if (numKeys == 1) {
    // Skip the numKeys (8 bytes) + first key (4 bytes) prefix to get the
    // roaring32 bitmap.
    constexpr std::size_t kRoaring32Offset =
        sizeof(uint64_t) + sizeof(uint32_t);
    auto roaring32 = std::string_view(payload).substr(kRoaring32Offset);
    if (is32bitBitmapNormalized(roaring32)) {
      buildBitmap(cudf::roaring_bitmap_type::BITS_32, roaring32, stream);
    } else {
      auto normalizedPayload = normalizeRoaring32(roaring32);
      buildBitmap(
          cudf::roaring_bitmap_type::BITS_32, normalizedPayload, stream);
    }
  } else {
    // Multiple keys. Use 64-bit dispatch
    if (is64bitBitmapNormalized(payload, numKeys)) {
      buildBitmap(cudf::roaring_bitmap_type::BITS_64, payload, stream);
    } else {
      auto normalizedPayload = normalizeRoaring64(payload, numKeys);
      buildBitmap(
          cudf::roaring_bitmap_type::BITS_64, normalizedPayload, stream);
    }
  }

  return;
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
