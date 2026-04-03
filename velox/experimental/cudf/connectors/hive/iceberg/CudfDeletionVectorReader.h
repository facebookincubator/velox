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
#include <string>
#include <unordered_map>

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
  struct BitmapImpl;

  CudfDeletionVectorReader(
      std::string filePath,
      uint64_t fileSizeInBytes,
      std::unordered_map<int32_t, std::string> lowerBounds,
      std::unordered_map<int32_t, std::string> upperBounds);

  ~CudfDeletionVectorReader();

  CudfDeletionVectorReader(CudfDeletionVectorReader&&) noexcept;
  CudfDeletionVectorReader& operator=(CudfDeletionVectorReader&&) noexcept;
  CudfDeletionVectorReader(const CudfDeletionVectorReader&) = delete;
  CudfDeletionVectorReader& operator=(const CudfDeletionVectorReader&) =
      delete;

  /// Reads the DV blob from the Puffin file, strips the DV-v1 envelope, and
  /// constructs the cuco roaring bitmap on the GPU.
  void loadAndInitialize(rmm::cuda_stream_view stream);

  /// Applies the deletion vector to a cudf table chunk.
  ///
  /// @param table The cudf table chunk to filter.
  /// @param startRow Absolute row index of the first row in this chunk.
  /// @param stream CUDA stream for kernel launches.
  /// @param mr Device memory resource for the output table.
  /// @return A new cudf table with deleted rows removed.
  std::unique_ptr<cudf::table> applyDeletionVector(
      cudf::table_view const& table,
      std::size_t startRow,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  static constexpr int32_t kDvOffsetFieldId = 100;
  static constexpr int32_t kDvLengthFieldId = 101;

 private:
  /// Reads raw bytes from the Puffin file. Defined in .cpp (uses Velox I/O).
  std::string loadBlob();

  /// Strips the DV-v1 envelope. Defined in .cpp (uses Velox macros).
  void parseDvBlobEnvelope();

  std::string filePath_;
  uint64_t fileSizeInBytes_;
  std::unordered_map<int32_t, std::string> lowerBounds_;
  std::unordered_map<int32_t, std::string> upperBounds_;

  std::string dvBlobBytes_;
  std::size_t dvPayloadOffset_{0};
  std::size_t dvPayloadSize_{0};

  std::unique_ptr<BitmapImpl> bitmap_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
