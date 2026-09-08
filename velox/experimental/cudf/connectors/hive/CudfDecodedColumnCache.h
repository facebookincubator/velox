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
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

class PinnedHostAllocation;
class PackedColumnCompression;

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
  size_t uncompressedPackedSize() const;
  bool compressed() const;
  const void* pinnedData() const;

 private:
  friend class CudfDecodedColumnCache;

  int64_t firstRow_;
  int64_t lastRow_;
  std::vector<uint8_t> metadata_;
  std::shared_ptr<const PinnedHostAllocation> data_;
  std::shared_ptr<const PackedColumnCompression> compression_;
};

struct CoveredColumnRange {
  std::shared_ptr<const PinnedColumnChunk> chunk;
  int64_t firstRow;
  int64_t lastRow;
};

/// Immutable Parquet metadata and derived row-group state shared by all cache
/// readers of one file. Keeping this state in the cache avoids rebuilding and
/// copying it for full-file, filter-free cache hits.
struct CachedParquetFileMetadata {
  std::shared_ptr<const cudf::io::parquet::FileMetaData> parquetMetadata;
  std::vector<int64_t> rowOffsets;
  std::vector<cudf::size_type> allRowGroups;
};

/// Experimental process-lifetime cache for decoded Parquet column ranges.
///
/// Decoded columns are packed into a CCCL CUDA pinned-memory pool. Entries are
/// immutable after publication and are never evicted. A request can be covered
/// by multiple overlapping entries as long as their union has no gaps.
class CudfDecodedColumnCache {
 public:
  static constexpr uint64_t kMaxPinnedBytes = 70ULL << 30;
  static constexpr uint64_t kMaxGpuBytes = 40ULL << 30;

  enum class CompressionMode {
    kNone,
    kColumn,
    kColumnAdvanced,
  };

  struct Stats {
    uint64_t maxPinnedBytes{0};
    uint64_t pinnedBytes{0};
    uint64_t insertedUncompressedBytes{0};
    uint64_t insertedStoredBytes{0};
    uint64_t insertedCompressedRanges{0};
    uint64_t insertedRawRanges{0};
    uint64_t compressionAttempts{0};
    uint64_t compressionEncodeNanos{0};
    uint64_t restoreCalls{0};
    uint64_t restoredStoredBytes{0};
    uint64_t restoredUncompressedBytes{0};
    uint64_t decompressionNanos{0};
    uint64_t pipelinedRestoreBatches{0};
    uint64_t maxGpuBytes{0};
    uint64_t gpuBytes{0};
    uint64_t gpuInsertedBytes{0};
    uint64_t gpuInsertedRanges{0};
    uint64_t gpuAdmissionRejectedRanges{0};
    uint64_t gpuRestoreCalls{0};
    uint64_t gpuRestoredBytes{0};
    uint64_t gpuRestoreBatches{0};
  };

  struct FileKey {
    std::string connectorId;
    std::string filePath;

    bool operator==(const FileKey&) const = default;
  };

  /// Identity for row-group selection derived from immutable Parquet metadata.
  /// filterKey is a deterministic serialization of the logical Velox filters;
  /// dynamic filters are not supported by the experimental cuDF connector.
  struct RowGroupSelectionKey {
    FileKey file;
    uint64_t splitStart;
    uint64_t splitSize;
    std::string filterKey;
    cudf::type_id timestampType;
    bool usePandasMetadata;
    bool useArrowSchema;
    bool allowMismatchedSchemas;

    bool operator==(const RowGroupSelectionKey&) const = default;
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

  /// All source-row intervals needed to assemble one output column. Keeping
  /// the intervals together lets cache restoration pipeline chunks across the
  /// complete file projection instead of materializing one column at a time.
  struct ColumnRangeRequest {
    ColumnKey key;
    std::vector<std::pair<int64_t, int64_t>> ranges;
  };

  using ParquetMetadataPtr =
      std::shared_ptr<const cudf::io::parquet::FileMetaData>;
  using MetadataPtr = std::shared_ptr<const CachedParquetFileMetadata>;
  using RowGroupSelectionPtr =
      std::shared_ptr<const std::vector<cudf::size_type>>;
  using ColumnRangePtr = std::shared_ptr<const PinnedColumnChunk>;

  static CudfDecodedColumnCache& instance();

  /// Overrides the process-lifetime pinned pool limit before instance() is
  /// first called. The experimental cache retains a 70 GiB default.
  static void configureMaxPinnedBytes(uint64_t maxPinnedBytes);

  /// Overrides the non-evicting GPU tier limit before instance() is first
  /// called. The tier remains unused unless the reader-level GPU cache option
  /// is enabled.
  static void configureMaxGpuBytes(uint64_t maxGpuBytes);

  static CompressionMode compressionModeFromString(std::string_view value);

  MetadataPtr findMetadata(const FileKey& key) const;
  MetadataPtr insertMetadataIfAbsent(
      FileKey key,
      ParquetMetadataPtr metadata);

  RowGroupSelectionPtr findRowGroupSelection(
      const RowGroupSelectionKey& key) const;
  RowGroupSelectionPtr insertRowGroupSelectionIfAbsent(
      RowGroupSelectionKey key,
      std::vector<cudf::size_type> rowGroups);

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
  /// already covered or when admitting it would exceed the configured pinned
  /// pool limit. Allocation failure is treated as non-admission, not query
  /// failure.
  bool insertColumnRangeIfAbsent(
      ColumnKey key,
      int64_t firstRow,
      int64_t lastRow,
      cudf::column_view column,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref tempMr,
      CompressionMode compressionMode,
      std::optional<rmm::device_async_resource_ref> gpuCacheMr =
          std::nullopt);

  /// Copies and inserts a decoded range into the non-evicting GPU tier.
  /// Returns false when the range is already covered, the tier is disabled,
  /// or admission would exceed the configured logical byte limit. Allocation
  /// failure is treated as non-admission, not query failure.
  bool insertGpuColumnRangeIfAbsent(
      ColumnKey key,
      int64_t firstRow,
      int64_t lastRow,
      cudf::column_view column,
      uint64_t estimatedBytes,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref cacheMr);

  /// Restores and concatenates [firstRow, lastRow) on the requested stream.
  /// Returns nullptr if the cache has a gap in the requested range.
  std::unique_ptr<cudf::column> materializeColumnRange(
      const ColumnKey& key,
      int64_t firstRow,
      int64_t lastRow,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref outputMr,
      rmm::device_async_resource_ref tempMr) const;

  /// Restores all requested columns with a two-slot file-level pipeline.
  /// H2D copies run on transferStream while the preceding chunk is decoded and
  /// materialized on stream. Returns nullopt if any requested interval has a
  /// cache gap; no GPU work is submitted until complete coverage is verified.
  std::optional<std::vector<std::unique_ptr<cudf::column>>>
  materializeColumnRanges(
      const std::vector<ColumnRangeRequest>& requests,
      rmm::cuda_stream_view stream,
      rmm::cuda_stream_view transferStream,
      rmm::device_async_resource_ref outputMr,
      rmm::device_async_resource_ref tempMr) const;

  /// Materializes each request directly from GPU-resident decoded chunks.
  /// The returned vector matches requests in size and contains nullptr for
  /// requests with any coverage gap, allowing callers to fall back to the CPU
  /// tier independently per column.
  std::vector<std::unique_ptr<cudf::column>> materializeGpuColumnRanges(
      const std::vector<ColumnRangeRequest>& requests,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref outputMr) const;

  uint64_t pinnedBytes() const;
  uint64_t maxPinnedBytes() const;
  uint64_t gpuBytes() const;
  uint64_t maxGpuBytes() const;
  Stats stats() const;

  /// Clears all entries for test isolation. Production code never calls this.
  void clearForTesting();

 private:
  struct GpuColumnChunk;
  struct Impl;

  CudfDecodedColumnCache();
  ~CudfDecodedColumnCache();

  std::unique_ptr<Impl> impl_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
