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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

// Forward declare instead of including the `IcebergDeleteFile.h` here since the
// header cannot be transitively included in the .cu file.
namespace facebook::velox::connector::hive::iceberg {
struct IcebergDeleteFile;
} // namespace facebook::velox::connector::hive::iceberg

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Reads and applies Iceberg V3 deletion vectors on the GPU.
///
/// Lifecycle:
///   1. Construct with the Puffin file path, size, and bounds maps extracted
///      from the IcebergDeleteFile.
///   2. Call loadAndInitialize() to read the blob from disk, parse the DV-v1
///      envelope, and build the cuco roaring bitmap on the GPU. The bitmap is
///      retained as a member so it is constructed only once per split.
///   3. Call applyDeletionVector() for each chunk read from the parquet reader
///      to filter out deleted rows.
class CudfDeletionVectorReader {
 public:
  //! Forward declaration of the opaque cuco roaring bitmap wrapper.
  struct RoaringBitmapImpl;

  CudfDeletionVectorReader(
      const velox::connector::hive::iceberg::IcebergDeleteFile& dvFile,
      uint64_t splitOffset = 0);

  ~CudfDeletionVectorReader();
  CudfDeletionVectorReader(CudfDeletionVectorReader&&) noexcept;

  /// Returns true when there is no more data. For DVs this is always true
  /// after `loadBitmap` has been called.
  bool noMoreData() const {
    return loaded_;
  }

  /// Applies the deletion vector to a cudf table chunk.
  ///
  /// @param table The cudf table chunk to filter.
  /// @param startRow Absolute row index of the first row in this chunk.
  /// @param rowMask Shared boolean mask buffer on device.
  /// @param stream CUDA stream for kernel launches.
  /// @param mr Device memory resource for the output table.
  /// @return A new cudf table with deleted rows removed.
  std::unique_ptr<cudf::table> applyDeletionVector(
      cudf::table_view const& table,
      std::size_t startRow,
      std::shared_ptr<rmm::device_buffer> rowMask,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  /// Field IDs used to encode DV blob offset and length in the
  /// IcebergDeleteFile bounds maps. The coordinator encodes these when
  /// building splits from Puffin file metadata.
  static constexpr int32_t kDvOffsetFieldId = 100;
  static constexpr int32_t kDvLengthFieldId = 101;

 private:
  /// Bitmap type for the deletion vector.
  enum class BitmapType : uint8_t { k32Bit = 0, k64Bit = 1 };

  /// Constructs cuco roaring bitmap from the normalized roaring bitmap
  /// payload.
  /// @tparam Bits Roaring bitmap type to build
  /// @param roaringBitmapPayload Normalized payload of the roaring bitmap.
  /// @param stream CUDA stream for bitmap construction
  template <BitmapType BitSize>
  void buildBitmap(
      std::string_view roaringBitmapPayload,
      rmm::cuda_stream_view stream);

  /// Loads the deletion vector blob from the Puffin file, strips the DV-v1
  /// envelope, and constructs the cuco roaring bitmap. Called lazily on the
  /// first `applyDeletionVector` call.
  void loadBitmap(rmm::cuda_stream_view stream);

  /// Deleter for RoaringBitmapImpl
  struct RoaringBitmapDeleter {
    void operator()(RoaringBitmapImpl* p) const;
  };

  /// Opaque wrapper class for cuco's 32 or 64 bit roaring bitmap
  std::unique_ptr<RoaringBitmapImpl, RoaringBitmapDeleter> bitmap_;

  /// Device buffer for the row indices vector.
  std::unique_ptr<rmm::device_buffer> rowIndices_;

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
