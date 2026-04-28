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

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

// Forward declare instead of including the `IcebergDeleteFile.h` here since the
// header cannot be transitively included in the .cu file.
namespace facebook::velox::connector::hive::iceberg {
struct IcebergDeleteFile;
} // namespace facebook::velox::connector::hive::iceberg

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Reads an Iceberg V3 deletion vector and applies it to the delete bitmap.
///
/// Supports both 64-bit Roaring64Bitmap format (used by Java's Roaring64Bitmap
/// at Meta's 500 PB scale) and legacy 32-bit RoaringBitmap format.
///
/// 64-bit format: [numGroups: uint64] then for each group:
///   [highBits: uint32] [32-bit RoaringBitmap in portable format]
///
/// 32-bit format: [cookie: uint32] [containerCount: uint32] ...
///   Detected by checking if the first 8 bytes match a 32-bit cookie (12346
///   or 12347).
class CudfDeletionVectorReader {
 public:
  CudfDeletionVectorReader(
      const velox::connector::hive::iceberg::IcebergDeleteFile& dvFile,
      uint64_t splitOffset = 0);

  ~CudfDeletionVectorReader() = default;
  CudfDeletionVectorReader(CudfDeletionVectorReader&&) noexcept = default;

  CudfDeletionVectorReader(const CudfDeletionVectorReader&) = delete;
  CudfDeletionVectorReader& operator=(const CudfDeletionVectorReader&) = delete;

  /// Returns true when there is no more data. For DVs this is always true
  /// after `loadBitmap` has been called.
  bool noMoreData() const {
    return loaded_;
  }

  /// Updates the deleted positions in the row mask in-place
  ///
  /// @param rowMask Mutable boolean mask column on device.
  /// @param startRow Absolute row index of the first row in this chunk.
  /// @param numRows Number of rows in the table chunk.
  /// @param stream CUDA stream for kernel launches.
  /// @param temp_mr Device memory resource for temporary allocations.
  void applyDeletes(
      cudf::mutable_column_view const& rowMask,
      std::size_t startRow,
      std::size_t numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref temp_mr);

  /// Field IDs used to encode DV blob offset and length in the
  /// IcebergDeleteFile bounds maps. The coordinator encodes these when
  /// building splits from Puffin file metadata.
  static constexpr int32_t kDvOffsetFieldId = 100;
  static constexpr int32_t kDvLengthFieldId = 101;

 private:
  /// Constructs cuco roaring bitmap from the normalized roaring bitmap
  /// payload.
  /// @param bitmapType Roaring bitmap type to build
  /// @param roaringBitmapPayload Normalized payload of the roaring bitmap.
  /// @param stream CUDA stream for bitmap construction
  void buildBitmap(
      cudf::roaring_bitmap_type bitmapType,
      std::string_view roaringBitmapPayload,
      rmm::cuda_stream_view stream);

  /// Loads the deletion vector blob from the Puffin file, strips the DV-v1
  /// envelope, and constructs the cuco roaring bitmap. Called lazily on the
  /// first `applyDeletionVector` call.
  void loadBitmap(rmm::cuda_stream_view stream);

  /// Opaque wrapper class for cuco's 32 or 64 bit roaring bitmap
  std::unique_ptr<cudf::roaring_bitmap> bitmap_;

  /// Row indices column
  std::unique_ptr<cudf::column> rowIndices_;

  /// Deletion vector file metadata.
  struct {
    const std::string filePath;
    const uint64_t fileSizeInBytes;
    const std::unordered_map<int32_t, std::string> lowerBounds;
    const std::unordered_map<int32_t, std::string> upperBounds;
  } dvFile_;
  uint64_t splitOffset_;

  /// Whether the bitmap has been loaded from the file.
  bool loaded_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
