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

#include <cudf/column/column.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

class PinnedHostAllocation;

/// An immutable decoded column range serialized into CUDA-pinned host memory.
class PinnedColumnChunk {
 public:
  int64_t firstRow() const {
    return firstRow_;
  }

  int64_t lastRow() const {
    return lastRow_;
  }

  size_t packedSize() const;
  const void* pinnedData() const;

 private:
  friend class CudfDecodedColumnCache;

  int64_t firstRow_;
  int64_t lastRow_;
  std::vector<uint8_t> metadata_;
  std::shared_ptr<const PinnedHostAllocation> data_;
};

struct CoveredColumnRange {
  std::shared_ptr<const PinnedColumnChunk> chunk;
  int64_t firstRow;
  int64_t lastRow;
};

/// Experimental process-lifetime cache for decoded Parquet column ranges.
///
/// Decoded columns are packed into a CCCL CUDA pinned-memory pool. Entries are
/// immutable after publication and are never evicted. A request can be covered
/// by multiple overlapping entries as long as their union has no gaps.
class CudfDecodedColumnCache {
 public:
  static constexpr uint64_t kMaxPinnedBytes = 70ULL << 30;

  struct FileKey {
    std::string connectorId;
    std::string filePath;

    bool operator==(const FileKey&) const = default;
  };

  /// Column identity excluding row range. Row ranges are stored independently
  /// so differently chunked scans can reuse one another.
  struct ColumnKey {
    FileKey file;
    int deviceId;
    std::string columnName;
    std::string veloxType;
    cudf::type_id timestampType;
    bool usePandasMetadata;
    bool useArrowSchema;
    bool allowMismatchedSchemas;

    bool operator==(const ColumnKey&) const = default;
  };

  using MetadataPtr = std::shared_ptr<const cudf::io::parquet::FileMetaData>;
  using ColumnRangePtr = std::shared_ptr<const PinnedColumnChunk>;

  static CudfDecodedColumnCache& instance();

  MetadataPtr findMetadata(const FileKey& key) const;
  MetadataPtr insertMetadataIfAbsent(FileKey key, MetadataPtr metadata);

  /// Returns a gap-free, ordered coverage of [firstRow, lastRow), or nullopt.
  std::optional<std::vector<CoveredColumnRange>> findColumnRanges(
      const ColumnKey& key,
      int64_t firstRow,
      int64_t lastRow) const;

  bool containsColumnRange(
      const ColumnKey& key,
      int64_t firstRow,
      int64_t lastRow) const;

  /// Packs and inserts [firstRow, lastRow). Returns false when the range is
  /// already covered or when admitting it would exceed the 70 GiB pinned pool
  /// limit. Allocation failure is treated as non-admission, not query failure.
  bool insertColumnRangeIfAbsent(
      ColumnKey key,
      int64_t firstRow,
      int64_t lastRow,
      cudf::column_view column,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref tempMr);

  /// Restores and concatenates [firstRow, lastRow) on the requested stream.
  /// Returns nullptr if the cache has a gap in the requested range.
  std::unique_ptr<cudf::column> materializeColumnRange(
      const ColumnKey& key,
      int64_t firstRow,
      int64_t lastRow,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref outputMr,
      rmm::device_async_resource_ref tempMr) const;

  uint64_t pinnedBytes() const;

  /// Clears all entries for test isolation. Production code never calls this.
  void clearForTesting();

 private:
  struct Impl;

  CudfDecodedColumnCache();
  ~CudfDecodedColumnCache();

  std::unique_ptr<Impl> impl_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
