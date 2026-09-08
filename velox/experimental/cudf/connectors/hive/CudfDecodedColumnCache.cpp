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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfDecodedColumnCache.h"
#include "velox/experimental/ucx-exchange/UcxColumnCodec.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/memory_pool>
#include <cuda_runtime_api.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <tuple>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

constexpr size_t kPackStagingBytes = 16ULL << 20;

template <typename T>
void hashCombine(size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct FileKeyHash {
  size_t operator()(const CudfDecodedColumnCache::FileKey& key) const {
    size_t seed = 0;
    hashCombine(seed, key.connectorId);
    hashCombine(seed, key.filePath);
    return seed;
  }
};

struct RowGroupSelectionKeyHash {
  size_t operator()(
      const CudfDecodedColumnCache::RowGroupSelectionKey& key) const {
    size_t seed = FileKeyHash{}(key.file);
    hashCombine(seed, key.splitStart);
    hashCombine(seed, key.splitSize);
    hashCombine(seed, key.filterKey);
    hashCombine(seed, static_cast<int>(key.timestampType));
    hashCombine(seed, key.usePandasMetadata);
    hashCombine(seed, key.useArrowSchema);
    hashCombine(seed, key.allowMismatchedSchemas);
    return seed;
  }
};

struct ColumnKeyHash {
  size_t operator()(const CudfDecodedColumnCache::ColumnKey& key) const {
    size_t seed = FileKeyHash{}(key.file);
    hashCombine(seed, key.deviceId);
    hashCombine(seed, key.columnName);
    hashCombine(seed, key.veloxType);
    hashCombine(seed, static_cast<int>(key.timestampType));
    hashCombine(seed, key.usePandasMetadata);
    hashCombine(seed, key.useArrowSchema);
    hashCombine(seed, key.allowMismatchedSchemas);
    return seed;
  }
};

cuda::memory_pool_properties pinnedPoolProperties(uint64_t maxPinnedBytes) {
  cuda::memory_pool_properties properties;
  properties.release_threshold = maxPinnedBytes;
  properties.max_pool_size = maxPinnedBytes;
  return properties;
}

struct CacheConfiguration {
  std::mutex mutex;
  uint64_t maxPinnedBytes{CudfDecodedColumnCache::kMaxPinnedBytes};
  uint64_t maxGpuBytes{CudfDecodedColumnCache::kMaxGpuBytes};
  bool initialized{false};
};

CacheConfiguration& cacheConfiguration() {
  static auto* configuration = new CacheConfiguration();
  return *configuration;
}

uint64_t takeConfiguredMaxPinnedBytes() {
  auto& configuration = cacheConfiguration();
  std::lock_guard<std::mutex> lock(configuration.mutex);
  configuration.initialized = true;
  return configuration.maxPinnedBytes;
}

uint64_t takeConfiguredMaxGpuBytes() {
  auto& configuration = cacheConfiguration();
  std::lock_guard<std::mutex> lock(configuration.mutex);
  return configuration.maxGpuBytes;
}

int currentNumaNode() {
  unsigned cpu = 0;
  unsigned node = 0;
  VELOX_CHECK_EQ(
      ::syscall(SYS_getcpu, &cpu, &node, nullptr),
      0,
      "Failed to determine the NUMA node for the decoded column cache");
  return static_cast<int>(node);
}

class CachePipelineEvent {
 public:
  CachePipelineEvent() {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~CachePipelineEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  CachePipelineEvent(const CachePipelineEvent&) = delete;
  CachePipelineEvent& operator=(const CachePipelineEvent&) = delete;

  void record(rmm::cuda_stream_view stream) const {
    CUDF_CUDA_TRY(cudaEventRecord(event_, stream.value()));
  }

  void wait(rmm::cuda_stream_view stream) const {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.value(), event_, 0));
  }

 private:
  cudaEvent_t event_{nullptr};
};

} // namespace

class PackedColumnCompression {
 public:
  PackedColumnCompression(
      std::vector<ucx_exchange::EncodedRegion> regions,
      size_t uncompressedBytes)
      : regions_(std::move(regions)), uncompressedBytes_(uncompressedBytes) {}

 private:
  friend class CudfDecodedColumnCache;
  friend class PinnedColumnChunk;

  std::vector<ucx_exchange::EncodedRegion> regions_;
  size_t uncompressedBytes_;
};

struct CudfDecodedColumnCache::GpuColumnChunk {
  int64_t firstRow;
  int64_t lastRow;
  uint64_t bytes;
  std::unique_ptr<cudf::column> column;
};

class PinnedHostAllocation {
 public:
  PinnedHostAllocation(
      cuda::pinned_memory_pool* pool,
      std::atomic<uint64_t>* allocatedBytes,
      void* data,
      size_t size)
      : pool_(pool),
        allocatedBytes_(allocatedBytes),
        data_(data),
        size_(size) {}

  ~PinnedHostAllocation() {
    if (data_ != nullptr) {
      pool_->deallocate_sync(data_, size_);
      allocatedBytes_->fetch_sub(size_, std::memory_order_relaxed);
    }
  }

  const void* data() const {
    return data_;
  }

  size_t size() const {
    return size_;
  }

 private:
  cuda::pinned_memory_pool* pool_;
  std::atomic<uint64_t>* allocatedBytes_;
  void* data_;
  size_t size_;
};

struct CudfDecodedColumnCache::Impl {
  Impl(uint64_t maxPinnedBytes, uint64_t maxGpuBytes)
      : maxPinnedBytes(maxPinnedBytes),
        maxGpuBytes(maxGpuBytes),
        pinnedPool(currentNumaNode(), pinnedPoolProperties(maxPinnedBytes)) {}

  std::shared_ptr<const PinnedHostAllocation> allocate(size_t size) {
    if (size == 0) {
      return std::make_shared<const PinnedHostAllocation>(
          &pinnedPool, &allocatedBytes, nullptr, 0);
    }

    auto current = allocatedBytes.load(std::memory_order_relaxed);
    do {
      if (size > maxPinnedBytes - current) {
        return nullptr;
      }
    } while (not allocatedBytes.compare_exchange_weak(
        current,
        current + size,
        std::memory_order_relaxed,
        std::memory_order_relaxed));

    try {
      auto* data = pinnedPool.allocate_sync(size);
      return std::make_shared<const PinnedHostAllocation>(
          &pinnedPool, &allocatedBytes, data, size);
    } catch (const std::exception& error) {
      allocatedBytes.fetch_sub(size, std::memory_order_relaxed);
      LOG(WARNING) << "Skipping decoded column cache admission after pinned "
                      "allocation failed: "
                   << error.what();
      return nullptr;
    }
  }

  std::optional<std::vector<CoveredColumnRange>> findColumnRangesLocked(
      const ColumnKey& key,
      int64_t firstRow,
      int64_t lastRow) const {
    const auto it = columns.find(key);
    if (it == columns.end()) {
      return std::nullopt;
    }

    std::vector<CoveredColumnRange> result;
    auto cursor = firstRow;
    while (cursor < lastRow) {
      ColumnRangePtr best;
      for (const auto& chunk : it->second) {
        if (chunk->firstRow() > cursor) {
          break;
        }
        if (chunk->lastRow() > cursor and
            (not best or chunk->lastRow() > best->lastRow())) {
          best = chunk;
        }
      }
      if (not best) {
        return std::nullopt;
      }

      const auto coveredUntil = std::min(lastRow, best->lastRow());
      result.push_back({best, cursor, coveredUntil});
      cursor = coveredUntil;
    }
    return result;
  }

  struct CoveredGpuColumnRange {
    std::shared_ptr<const GpuColumnChunk> chunk;
    int64_t firstRow;
    int64_t lastRow;
  };

  std::optional<std::vector<CoveredGpuColumnRange>>
  findGpuColumnRangesLocked(
      const ColumnKey& key,
      int64_t firstRow,
      int64_t lastRow) const {
    const auto it = gpuColumns.find(key);
    if (it == gpuColumns.end()) {
      return std::nullopt;
    }

    std::vector<CoveredGpuColumnRange> result;
    auto cursor = firstRow;
    while (cursor < lastRow) {
      std::shared_ptr<const GpuColumnChunk> best;
      for (const auto& chunk : it->second) {
        if (chunk->firstRow > cursor) {
          break;
        }
        if (chunk->lastRow > cursor and
            (not best or chunk->lastRow > best->lastRow)) {
          best = chunk;
        }
      }
      if (not best) {
        return std::nullopt;
      }

      const auto coveredUntil = std::min(lastRow, best->lastRow);
      result.push_back({best, cursor, coveredUntil});
      cursor = coveredUntil;
    }
    return result;
  }

  bool reserveGpuBytes(uint64_t size) {
    if (maxGpuBytes == 0 or size > maxGpuBytes) {
      return false;
    }
    auto current = gpuBytes.load(std::memory_order_relaxed);
    do {
      if (size > maxGpuBytes - current) {
        return false;
      }
    } while (not gpuBytes.compare_exchange_weak(
        current,
        current + size,
        std::memory_order_relaxed,
        std::memory_order_relaxed));
    return true;
  }

  mutable std::mutex mutex;
  const uint64_t maxPinnedBytes;
  const uint64_t maxGpuBytes;
  cuda::pinned_memory_pool pinnedPool;
  std::atomic<uint64_t> allocatedBytes{0};
  std::atomic<uint64_t> insertedUncompressedBytes{0};
  std::atomic<uint64_t> insertedStoredBytes{0};
  std::atomic<uint64_t> insertedCompressedRanges{0};
  std::atomic<uint64_t> insertedRawRanges{0};
  std::atomic<uint64_t> compressionAttempts{0};
  std::atomic<uint64_t> compressionEncodeNanos{0};
  std::atomic<uint64_t> restoreCalls{0};
  std::atomic<uint64_t> restoredStoredBytes{0};
  std::atomic<uint64_t> restoredUncompressedBytes{0};
  std::atomic<uint64_t> decompressionNanos{0};
  std::atomic<uint64_t> pipelinedRestoreBatches{0};
  std::atomic<uint64_t> gpuBytes{0};
  std::atomic<uint64_t> gpuInsertedBytes{0};
  std::atomic<uint64_t> gpuInsertedRanges{0};
  std::atomic<uint64_t> gpuAdmissionRejectedRanges{0};
  std::atomic<uint64_t> gpuRestoreCalls{0};
  std::atomic<uint64_t> gpuRestoredBytes{0};
  std::atomic<uint64_t> gpuRestoreBatches{0};
  std::unordered_map<FileKey, MetadataPtr, FileKeyHash> metadata;
  std::unordered_map<
      RowGroupSelectionKey,
      RowGroupSelectionPtr,
      RowGroupSelectionKeyHash>
      rowGroupSelections;
  std::unordered_map<ColumnKey, std::vector<ColumnRangePtr>, ColumnKeyHash>
      columns;
  std::unordered_map<
      ColumnKey,
      std::vector<std::shared_ptr<const GpuColumnChunk>>,
      ColumnKeyHash>
      gpuColumns;
};

size_t PinnedColumnChunk::packedSize() const {
  return data_->size();
}

size_t PinnedColumnChunk::uncompressedPackedSize() const {
  return compression_ ? compression_->uncompressedBytes_ : packedSize();
}

bool PinnedColumnChunk::compressed() const {
  return compression_ != nullptr;
}

const void* PinnedColumnChunk::pinnedData() const {
  return data_->data();
}

CudfDecodedColumnCache::CudfDecodedColumnCache()
    : impl_(std::make_unique<Impl>(
          takeConfiguredMaxPinnedBytes(), takeConfiguredMaxGpuBytes())) {}

CudfDecodedColumnCache::~CudfDecodedColumnCache() = default;

CudfDecodedColumnCache& CudfDecodedColumnCache::instance() {
  // Intentionally process-lifetime: avoids CUDA pool destruction during static
  // teardown and implements the prototype's non-evicting lifetime.
  static auto* cache = new CudfDecodedColumnCache();
  return *cache;
}

void CudfDecodedColumnCache::configureMaxPinnedBytes(
    uint64_t maxPinnedBytes) {
  VELOX_USER_CHECK_GT(
      maxPinnedBytes, 0, "Decoded column cache limit must be positive");
  auto& configuration = cacheConfiguration();
  std::lock_guard<std::mutex> lock(configuration.mutex);
  VELOX_USER_CHECK(
      not configuration.initialized,
      "Decoded column cache limit must be configured before first use");
  configuration.maxPinnedBytes = maxPinnedBytes;
}

void CudfDecodedColumnCache::configureMaxGpuBytes(uint64_t maxGpuBytes) {
  auto& configuration = cacheConfiguration();
  std::lock_guard<std::mutex> lock(configuration.mutex);
  VELOX_USER_CHECK(
      not configuration.initialized,
      "Decoded column GPU cache limit must be configured before first use");
  configuration.maxGpuBytes = maxGpuBytes;
}

CudfDecodedColumnCache::CompressionMode
CudfDecodedColumnCache::compressionModeFromString(std::string_view value) {
  if (value == "none") {
    return CompressionMode::kNone;
  }
  if (value == "column") {
    return CompressionMode::kColumn;
  }
  VELOX_USER_CHECK_EQ(
      value,
      "column-advanced",
      "Unsupported decoded column cache compression '{}'. Expected none, column, or column-advanced",
      value);
  return CompressionMode::kColumnAdvanced;
}

CudfDecodedColumnCache::MetadataPtr CudfDecodedColumnCache::findMetadata(
    const FileKey& key) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto it = impl_->metadata.find(key);
  return it == impl_->metadata.end() ? nullptr : it->second;
}

CudfDecodedColumnCache::MetadataPtr
CudfDecodedColumnCache::insertMetadataIfAbsent(
    FileKey key,
    ParquetMetadataPtr metadata) {
  VELOX_CHECK_NOT_NULL(metadata);
  auto candidate = std::make_shared<CachedParquetFileMetadata>();
  candidate->parquetMetadata = std::move(metadata);
  const auto numRowGroups = candidate->parquetMetadata->row_groups.size();
  candidate->rowOffsets.reserve(numRowGroups + 1);
  candidate->rowOffsets.push_back(0);
  for (const auto& rowGroup : candidate->parquetMetadata->row_groups) {
    VELOX_CHECK_GE(rowGroup.num_rows, 0);
    candidate->rowOffsets.push_back(
        candidate->rowOffsets.back() + rowGroup.num_rows);
  }
  candidate->allRowGroups.resize(numRowGroups);
  std::iota(
      candidate->allRowGroups.begin(), candidate->allRowGroups.end(), 0);

  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->metadata.try_emplace(std::move(key), std::move(candidate))
      .first->second;
}

CudfDecodedColumnCache::RowGroupSelectionPtr
CudfDecodedColumnCache::findRowGroupSelection(
    const RowGroupSelectionKey& key) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto it = impl_->rowGroupSelections.find(key);
  return it == impl_->rowGroupSelections.end() ? nullptr : it->second;
}

CudfDecodedColumnCache::RowGroupSelectionPtr
CudfDecodedColumnCache::insertRowGroupSelectionIfAbsent(
    RowGroupSelectionKey key,
    std::vector<cudf::size_type> rowGroups) {
  auto candidate = std::make_shared<const std::vector<cudf::size_type>>(
      std::move(rowGroups));
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->rowGroupSelections
      .try_emplace(std::move(key), std::move(candidate))
      .first->second;
}

std::optional<std::vector<CoveredColumnRange>>
CudfDecodedColumnCache::findColumnRanges(
    const ColumnKey& key,
    int64_t firstRow,
    int64_t lastRow) const {
  VELOX_CHECK_LT(firstRow, lastRow, "Decoded cache range must be non-empty");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->findColumnRangesLocked(key, firstRow, lastRow);
}

bool CudfDecodedColumnCache::containsColumnRange(
    const ColumnKey& key,
    int64_t firstRow,
    int64_t lastRow) const {
  return findColumnRanges(key, firstRow, lastRow).has_value();
}

bool CudfDecodedColumnCache::insertColumnRangeIfAbsent(
    ColumnKey key,
    int64_t firstRow,
    int64_t lastRow,
    cudf::column_view column,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref tempMr,
    CompressionMode compressionMode,
    std::optional<rmm::device_async_resource_ref> gpuCacheMr) {
  VELOX_CHECK_LT(firstRow, lastRow, "Decoded cache range must be non-empty");
  VELOX_CHECK_EQ(
      lastRow - firstRow,
      column.size(),
      "Decoded cache range must match column size");
  if (containsColumnRange(key, firstRow, lastRow)) {
    return false;
  }

  const std::vector<cudf::column_view> columns{column};
  const auto table = cudf::table_view{columns};
  std::vector<uint8_t> metadata;
  std::shared_ptr<const PinnedHostAllocation> pinnedData;
  std::shared_ptr<const PackedColumnCompression> compression;
  size_t uncompressedPackedSize{0};
  uint64_t compressionEncodeNanos{0};
  bool compressionAttempted{false};

  const auto packRawChunked = [&]() -> bool {
    auto packer =
        cudf::chunked_pack::create(table, kPackStagingBytes, stream, tempMr);
    uncompressedPackedSize = packer->get_total_contiguous_size();
    pinnedData = impl_->allocate(uncompressedPackedSize);
    if (not pinnedData) {
      return false;
    }

    rmm::device_buffer staging(kPackStagingBytes, stream, tempMr);
    auto* destination =
        const_cast<uint8_t*>(static_cast<const uint8_t*>(pinnedData->data()));
    size_t offset = 0;
    while (packer->has_next()) {
      const auto bytes = packer->next(
          cudf::device_span<uint8_t>{
              static_cast<uint8_t*>(staging.data()), staging.size()});
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          destination + offset,
          staging.data(),
          bytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
      offset += bytes;
    }
    VELOX_CHECK_EQ(offset, uncompressedPackedSize);
    metadata = std::move(*packer->build_metadata());
    stream.synchronize();
    return true;
  };

  if (compressionMode == CompressionMode::kNone) {
    if (not packRawChunked()) {
      return false;
    }
  } else {
    try {
      auto packed = cudf::pack(table, stream, tempMr);
      uncompressedPackedSize = packed.gpu_data->size();
      compressionAttempted = uncompressedPackedSize > 0;

      ucx_exchange::PackedCompressResult compressed;
      if (compressionAttempted) {
        const auto start = std::chrono::steady_clock::now();
        compressed = ucx_exchange::compressPacked(
            packed.metadata->data(),
            packed.gpu_data->data(),
            packed.gpu_data->size(),
            stream,
            0.02,
            compressionMode == CompressionMode::kColumnAdvanced,
            0);
        compressionEncodeNanos =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
      }

      const auto* storedData =
          compressed.used ? compressed.data.data() : packed.gpu_data->data();
      const auto storedSize =
          compressed.used ? compressed.data.size() : packed.gpu_data->size();
      pinnedData = impl_->allocate(storedSize);
      if (not pinnedData) {
        return false;
      }
      if (storedSize > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            const_cast<void*>(pinnedData->data()),
            storedData,
            storedSize,
            cudaMemcpyDeviceToHost,
            stream.value()));
      }
      metadata = std::move(*packed.metadata);
      if (compressed.used) {
        compression = std::make_shared<const PackedColumnCompression>(
            std::move(compressed.regions), uncompressedPackedSize);
      }
      stream.synchronize();
    } catch (const std::exception& error) {
      LOG(WARNING)
          << "Decoded column cache compression failed; storing the range raw: "
          << error.what();
      compressionAttempted = true;
      compression.reset();
      pinnedData.reset();
      metadata.clear();
      uncompressedPackedSize = 0;
      if (not packRawChunked()) {
        return false;
      }
    }
  }

  auto candidate = std::make_shared<PinnedColumnChunk>();
  candidate->firstRow_ = firstRow;
  candidate->lastRow_ = lastRow;
  candidate->metadata_ = std::move(metadata);
  candidate->data_ = std::move(pinnedData);
  candidate->compression_ = std::move(compression);

  auto gpuKey = key;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->findColumnRangesLocked(key, firstRow, lastRow).has_value()) {
      return false;
    }
    const auto storedSize = candidate->packedSize();
    const auto isCompressed = candidate->compressed();
    auto& chunks = impl_->columns[std::move(key)];
    chunks.push_back(std::move(candidate));
    std::sort(
        chunks.begin(), chunks.end(), [](const auto& left, const auto& right) {
          return std::tie(left->firstRow_, left->lastRow_) <
              std::tie(right->firstRow_, right->lastRow_);
        });
    impl_->insertedUncompressedBytes.fetch_add(
        uncompressedPackedSize, std::memory_order_relaxed);
    impl_->insertedStoredBytes.fetch_add(storedSize, std::memory_order_relaxed);
    (isCompressed ? impl_->insertedCompressedRanges
                  : impl_->insertedRawRanges)
        .fetch_add(1, std::memory_order_relaxed);
    if (compressionAttempted) {
      impl_->compressionAttempts.fetch_add(1, std::memory_order_relaxed);
      impl_->compressionEncodeNanos.fetch_add(
          compressionEncodeNanos, std::memory_order_relaxed);
    }
  }
  if (gpuCacheMr.has_value()) {
    insertGpuColumnRangeIfAbsent(
        std::move(gpuKey),
        firstRow,
        lastRow,
        column,
        uncompressedPackedSize,
        stream,
        gpuCacheMr.value());
  }
  return true;
}

bool CudfDecodedColumnCache::insertGpuColumnRangeIfAbsent(
    ColumnKey key,
    int64_t firstRow,
    int64_t lastRow,
    cudf::column_view column,
    uint64_t estimatedBytes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref cacheMr) {
  VELOX_CHECK_LT(firstRow, lastRow, "Decoded GPU cache range must be non-empty");
  VELOX_CHECK_EQ(
      lastRow - firstRow,
      column.size(),
      "Decoded GPU cache range must match column size");
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->findGpuColumnRangesLocked(key, firstRow, lastRow).has_value()) {
      return false;
    }
  }

  estimatedBytes = std::max<uint64_t>(estimatedBytes, 1);
  if (not impl_->reserveGpuBytes(estimatedBytes)) {
    impl_->gpuAdmissionRejectedRanges.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  std::unique_ptr<cudf::column> deviceColumn;
  uint64_t reservedBytes = estimatedBytes;
  try {
    deviceColumn =
        std::make_unique<cudf::column>(column, stream, cacheMr);
    const auto actualBytes = deviceColumn->alloc_size();
    if (actualBytes > reservedBytes) {
      const auto additionalBytes = actualBytes - reservedBytes;
      if (not impl_->reserveGpuBytes(additionalBytes)) {
        impl_->gpuBytes.fetch_sub(reservedBytes, std::memory_order_relaxed);
        impl_->gpuAdmissionRejectedRanges.fetch_add(
            1, std::memory_order_relaxed);
        return false;
      }
    } else if (actualBytes < reservedBytes) {
      impl_->gpuBytes.fetch_sub(
          reservedBytes - actualBytes, std::memory_order_relaxed);
    }
    reservedBytes = actualBytes;
    stream.synchronize();
  } catch (const std::exception& error) {
    impl_->gpuBytes.fetch_sub(reservedBytes, std::memory_order_relaxed);
    impl_->gpuAdmissionRejectedRanges.fetch_add(1, std::memory_order_relaxed);
    LOG(WARNING) << "Skipping decoded column GPU cache admission after device "
                    "allocation failed: "
                 << error.what();
    return false;
  }

  auto candidate = std::make_shared<GpuColumnChunk>(GpuColumnChunk{
      .firstRow = firstRow,
      .lastRow = lastRow,
      .bytes = reservedBytes,
      .column = std::move(deviceColumn),
  });

  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->findGpuColumnRangesLocked(key, firstRow, lastRow).has_value()) {
    impl_->gpuBytes.fetch_sub(reservedBytes, std::memory_order_relaxed);
    return false;
  }
  auto& chunks = impl_->gpuColumns[std::move(key)];
  chunks.push_back(std::move(candidate));
  std::sort(
      chunks.begin(), chunks.end(), [](const auto& left, const auto& right) {
        return std::tie(left->firstRow, left->lastRow) <
            std::tie(right->firstRow, right->lastRow);
      });
  impl_->gpuInsertedBytes.fetch_add(reservedBytes, std::memory_order_relaxed);
  impl_->gpuInsertedRanges.fetch_add(1, std::memory_order_relaxed);
  return true;
}

std::unique_ptr<cudf::column> CudfDecodedColumnCache::materializeColumnRange(
    const ColumnKey& key,
    int64_t firstRow,
    int64_t lastRow,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref outputMr,
    rmm::device_async_resource_ref tempMr) const {
  auto coverage = findColumnRanges(key, firstRow, lastRow);
  if (not coverage) {
    return nullptr;
  }

  std::vector<std::unique_ptr<cudf::column>> pieces;
  pieces.reserve(coverage->size());
  for (const auto& range : *coverage) {
    const auto& chunk = range.chunk;
    rmm::device_buffer storedData(chunk->packedSize(), stream, tempMr);
    if (chunk->packedSize() > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          storedData.data(),
          chunk->pinnedData(),
          chunk->packedSize(),
          cudaMemcpyHostToDevice,
          stream.value()));
    }
    rmm::device_buffer decompressedData;
    const uint8_t* packedData = static_cast<const uint8_t*>(storedData.data());
    if (chunk->compressed()) {
      const auto start = std::chrono::steady_clock::now();
      decompressedData = ucx_exchange::decompressPacked(
          storedData.data(),
          chunk->compression_->regions_,
          chunk->compression_->uncompressedBytes_,
          stream);
      impl_->decompressionNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count(),
          std::memory_order_relaxed);
      packedData = static_cast<const uint8_t*>(decompressedData.data());
    }
    impl_->restoreCalls.fetch_add(1, std::memory_order_relaxed);
    impl_->restoredStoredBytes.fetch_add(
        chunk->packedSize(), std::memory_order_relaxed);
    impl_->restoredUncompressedBytes.fetch_add(
        chunk->uncompressedPackedSize(), std::memory_order_relaxed);
    const auto unpacked = cudf::unpack(chunk->metadata_.data(), packedData);
    VELOX_CHECK_EQ(unpacked.num_columns(), 1);

    const auto relativeFirst = range.firstRow - chunk->firstRow();
    const auto relativeLast = range.lastRow - chunk->firstRow();
    VELOX_CHECK_LE(
        relativeLast,
        static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()));
    const auto slice = cudf::slice(
        unpacked,
        {static_cast<cudf::size_type>(relativeFirst),
         static_cast<cudf::size_type>(relativeLast)},
        stream);

    if (coverage->size() == 1) {
      return std::make_unique<cudf::column>(
          slice.front().column(0), stream, outputMr);
    }
    pieces.push_back(
        std::make_unique<cudf::column>(
            slice.front().column(0), stream, tempMr));
  }

  std::vector<cudf::column_view> pieceViews;
  pieceViews.reserve(pieces.size());
  for (const auto& piece : pieces) {
    pieceViews.push_back(piece->view());
  }
  return cudf::concatenate(pieceViews, stream, outputMr);
}

std::optional<std::vector<std::unique_ptr<cudf::column>>>
CudfDecodedColumnCache::materializeColumnRanges(
    const std::vector<ColumnRangeRequest>& requests,
    rmm::cuda_stream_view stream,
    rmm::cuda_stream_view transferStream,
    rmm::device_async_resource_ref outputMr,
    rmm::device_async_resource_ref tempMr) const {
  struct WorkItem {
    size_t requestIndex;
    CoveredColumnRange range;
  };

  std::vector<WorkItem> work;
  std::vector<size_t> pieceCounts(requests.size(), 0);
  for (size_t requestIndex = 0; requestIndex < requests.size();
       ++requestIndex) {
    for (const auto& [firstRow, lastRow] : requests[requestIndex].ranges) {
      auto coverage = findColumnRanges(
          requests[requestIndex].key, firstRow, lastRow);
      if (not coverage) {
        return std::nullopt;
      }
      pieceCounts[requestIndex] += coverage->size();
      for (auto& range : *coverage) {
        work.push_back({requestIndex, std::move(range)});
      }
    }
  }

  std::vector<std::vector<std::unique_ptr<cudf::column>>> pieces(
      requests.size());
  for (size_t requestIndex = 0; requestIndex < requests.size();
       ++requestIndex) {
    pieces[requestIndex].reserve(pieceCounts[requestIndex]);
  }
  if (work.empty()) {
    return std::vector<std::unique_ptr<cudf::column>>{};
  }

  struct TransferSlot {
    rmm::device_buffer storedData;
    CachePipelineEvent ready;
    CachePipelineEvent consumed;
    bool hasPendingConsumer{false};
  };
  std::array<TransferSlot, 2> slots;

  const auto stage = [&](size_t workIndex) {
    auto& slot = slots[workIndex % slots.size()];
    if (slot.hasPendingConsumer) {
      slot.consumed.wait(transferStream);
    }
    const auto& chunk = work[workIndex].range.chunk;
    slot.storedData =
        rmm::device_buffer(chunk->packedSize(), transferStream, tempMr);
    if (chunk->packedSize() > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          slot.storedData.data(),
          chunk->pinnedData(),
          chunk->packedSize(),
          cudaMemcpyHostToDevice,
          transferStream.value()));
    }
    slot.ready.record(transferStream);
    slot.hasPendingConsumer = false;
  };

  stage(0);
  for (size_t workIndex = 0; workIndex < work.size(); ++workIndex) {
    if (workIndex + 1 < work.size()) {
      stage(workIndex + 1);
    }

    auto& slot = slots[workIndex % slots.size()];
    const auto& item = work[workIndex];
    const auto& chunk = item.range.chunk;
    slot.ready.wait(stream);

    rmm::device_buffer decompressedData;
    const uint8_t* packedData =
        static_cast<const uint8_t*>(slot.storedData.data());
    if (chunk->compressed()) {
      const auto start = std::chrono::steady_clock::now();
      decompressedData = ucx_exchange::decompressPacked(
          slot.storedData.data(),
          chunk->compression_->regions_,
          chunk->compression_->uncompressedBytes_,
          stream);
      impl_->decompressionNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count(),
          std::memory_order_relaxed);
      packedData = static_cast<const uint8_t*>(decompressedData.data());
    }

    impl_->restoreCalls.fetch_add(1, std::memory_order_relaxed);
    impl_->restoredStoredBytes.fetch_add(
        chunk->packedSize(), std::memory_order_relaxed);
    impl_->restoredUncompressedBytes.fetch_add(
        chunk->uncompressedPackedSize(), std::memory_order_relaxed);

    const auto unpacked = cudf::unpack(chunk->metadata_.data(), packedData);
    VELOX_CHECK_EQ(unpacked.num_columns(), 1);
    const auto relativeFirst = item.range.firstRow - chunk->firstRow();
    const auto relativeLast = item.range.lastRow - chunk->firstRow();
    VELOX_CHECK_LE(
        relativeLast,
        static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()));
    const auto slice = cudf::slice(
        unpacked,
        {static_cast<cudf::size_type>(relativeFirst),
         static_cast<cudf::size_type>(relativeLast)},
        stream);
    auto piece = std::make_unique<cudf::column>(
        slice.front().column(0),
        stream,
        pieceCounts[item.requestIndex] == 1 ? outputMr : tempMr);
    pieces[item.requestIndex].push_back(std::move(piece));
    slot.consumed.record(stream);
    slot.hasPendingConsumer = true;
  }

  // Order the staging-buffer releases after their final consumers without a
  // device-wide or host-side synchronization.
  for (auto& slot : slots) {
    if (slot.hasPendingConsumer) {
      slot.consumed.wait(transferStream);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> outputs;
  outputs.reserve(requests.size());
  for (auto& requestPieces : pieces) {
    VELOX_CHECK(not requestPieces.empty());
    if (requestPieces.size() == 1) {
      outputs.push_back(std::move(requestPieces.front()));
      continue;
    }
    std::vector<cudf::column_view> pieceViews;
    pieceViews.reserve(requestPieces.size());
    for (const auto& piece : requestPieces) {
      pieceViews.push_back(piece->view());
    }
    outputs.push_back(cudf::concatenate(pieceViews, stream, outputMr));
  }
  impl_->pipelinedRestoreBatches.fetch_add(1, std::memory_order_relaxed);
  return outputs;
}

std::vector<std::unique_ptr<cudf::column>>
CudfDecodedColumnCache::materializeGpuColumnRanges(
    const std::vector<ColumnRangeRequest>& requests,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref outputMr) const {
  std::vector<std::unique_ptr<cudf::column>> outputs(requests.size());
  bool restoredAny = false;
  for (size_t requestIndex = 0; requestIndex < requests.size();
       ++requestIndex) {
    std::vector<Impl::CoveredGpuColumnRange> coverage;
    bool fullyCovered = true;
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      for (const auto& [firstRow, lastRow] : requests[requestIndex].ranges) {
        auto rangeCoverage = impl_->findGpuColumnRangesLocked(
            requests[requestIndex].key, firstRow, lastRow);
        if (not rangeCoverage) {
          fullyCovered = false;
          break;
        }
        coverage.insert(
            coverage.end(),
            std::make_move_iterator(rangeCoverage->begin()),
            std::make_move_iterator(rangeCoverage->end()));
      }
    }
    if (not fullyCovered or coverage.empty()) {
      continue;
    }

    std::vector<cudf::column_view> pieceViews;
    pieceViews.reserve(coverage.size());
    for (const auto& range : coverage) {
      const auto relativeFirst = range.firstRow - range.chunk->firstRow;
      const auto relativeLast = range.lastRow - range.chunk->firstRow;
      VELOX_CHECK_LE(
          relativeLast,
          static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()));
      const auto slices = cudf::slice(
          range.chunk->column->view(),
          {static_cast<cudf::size_type>(relativeFirst),
           static_cast<cudf::size_type>(relativeLast)},
          stream);
      VELOX_CHECK_EQ(slices.size(), 1);
      pieceViews.push_back(slices.front());
    }

    auto output = pieceViews.size() == 1
        ? std::make_unique<cudf::column>(pieceViews.front(), stream, outputMr)
        : cudf::concatenate(pieceViews, stream, outputMr);
    impl_->gpuRestoreCalls.fetch_add(
        coverage.size(), std::memory_order_relaxed);
    impl_->gpuRestoredBytes.fetch_add(
        output->alloc_size(), std::memory_order_relaxed);
    outputs[requestIndex] = std::move(output);
    restoredAny = true;
  }
  if (restoredAny) {
    impl_->gpuRestoreBatches.fetch_add(1, std::memory_order_relaxed);
  }
  return outputs;
}

uint64_t CudfDecodedColumnCache::pinnedBytes() const {
  return impl_->allocatedBytes.load(std::memory_order_relaxed);
}

uint64_t CudfDecodedColumnCache::maxPinnedBytes() const {
  return impl_->maxPinnedBytes;
}

uint64_t CudfDecodedColumnCache::gpuBytes() const {
  return impl_->gpuBytes.load(std::memory_order_relaxed);
}

uint64_t CudfDecodedColumnCache::maxGpuBytes() const {
  return impl_->maxGpuBytes;
}

CudfDecodedColumnCache::Stats CudfDecodedColumnCache::stats() const {
  return {
      .maxPinnedBytes = impl_->maxPinnedBytes,
      .pinnedBytes = impl_->allocatedBytes.load(std::memory_order_relaxed),
      .insertedUncompressedBytes =
          impl_->insertedUncompressedBytes.load(std::memory_order_relaxed),
      .insertedStoredBytes =
          impl_->insertedStoredBytes.load(std::memory_order_relaxed),
      .insertedCompressedRanges =
          impl_->insertedCompressedRanges.load(std::memory_order_relaxed),
      .insertedRawRanges =
          impl_->insertedRawRanges.load(std::memory_order_relaxed),
      .compressionAttempts =
          impl_->compressionAttempts.load(std::memory_order_relaxed),
      .compressionEncodeNanos =
          impl_->compressionEncodeNanos.load(std::memory_order_relaxed),
      .restoreCalls = impl_->restoreCalls.load(std::memory_order_relaxed),
      .restoredStoredBytes =
          impl_->restoredStoredBytes.load(std::memory_order_relaxed),
      .restoredUncompressedBytes =
          impl_->restoredUncompressedBytes.load(std::memory_order_relaxed),
      .decompressionNanos =
          impl_->decompressionNanos.load(std::memory_order_relaxed),
      .pipelinedRestoreBatches =
          impl_->pipelinedRestoreBatches.load(std::memory_order_relaxed),
      .maxGpuBytes = impl_->maxGpuBytes,
      .gpuBytes = impl_->gpuBytes.load(std::memory_order_relaxed),
      .gpuInsertedBytes =
          impl_->gpuInsertedBytes.load(std::memory_order_relaxed),
      .gpuInsertedRanges =
          impl_->gpuInsertedRanges.load(std::memory_order_relaxed),
      .gpuAdmissionRejectedRanges =
          impl_->gpuAdmissionRejectedRanges.load(std::memory_order_relaxed),
      .gpuRestoreCalls =
          impl_->gpuRestoreCalls.load(std::memory_order_relaxed),
      .gpuRestoredBytes =
          impl_->gpuRestoredBytes.load(std::memory_order_relaxed),
      .gpuRestoreBatches =
          impl_->gpuRestoreBatches.load(std::memory_order_relaxed),
  };
}

void CudfDecodedColumnCache::clearForTesting() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->columns.clear();
  impl_->gpuColumns.clear();
  impl_->metadata.clear();
  impl_->rowGroupSelections.clear();
  impl_->insertedUncompressedBytes.store(0, std::memory_order_relaxed);
  impl_->insertedStoredBytes.store(0, std::memory_order_relaxed);
  impl_->insertedCompressedRanges.store(0, std::memory_order_relaxed);
  impl_->insertedRawRanges.store(0, std::memory_order_relaxed);
  impl_->compressionAttempts.store(0, std::memory_order_relaxed);
  impl_->compressionEncodeNanos.store(0, std::memory_order_relaxed);
  impl_->restoreCalls.store(0, std::memory_order_relaxed);
  impl_->restoredStoredBytes.store(0, std::memory_order_relaxed);
  impl_->restoredUncompressedBytes.store(0, std::memory_order_relaxed);
  impl_->decompressionNanos.store(0, std::memory_order_relaxed);
  impl_->pipelinedRestoreBatches.store(0, std::memory_order_relaxed);
  impl_->gpuBytes.store(0, std::memory_order_relaxed);
  impl_->gpuInsertedBytes.store(0, std::memory_order_relaxed);
  impl_->gpuInsertedRanges.store(0, std::memory_order_relaxed);
  impl_->gpuAdmissionRejectedRanges.store(0, std::memory_order_relaxed);
  impl_->gpuRestoreCalls.store(0, std::memory_order_relaxed);
  impl_->gpuRestoredBytes.store(0, std::memory_order_relaxed);
  impl_->gpuRestoreBatches.store(0, std::memory_order_relaxed);
}

} // namespace facebook::velox::cudf_velox::connector::hive
