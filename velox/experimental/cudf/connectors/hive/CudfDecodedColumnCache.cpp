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
#include <atomic>
#include <chrono>
#include <functional>
#include <limits>
#include <mutex>
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

cuda::memory_pool_properties pinnedPoolProperties() {
  cuda::memory_pool_properties properties;
  properties.release_threshold = CudfDecodedColumnCache::kMaxPinnedBytes;
  properties.max_pool_size = CudfDecodedColumnCache::kMaxPinnedBytes;
  return properties;
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
  Impl() : pinnedPool(currentNumaNode(), pinnedPoolProperties()) {}

  std::shared_ptr<const PinnedHostAllocation> allocate(size_t size) {
    if (size == 0) {
      return std::make_shared<const PinnedHostAllocation>(
          &pinnedPool, &allocatedBytes, nullptr, 0);
    }

    auto current = allocatedBytes.load(std::memory_order_relaxed);
    do {
      if (size > kMaxPinnedBytes - current) {
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

  mutable std::mutex mutex;
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
  std::unordered_map<FileKey, MetadataPtr, FileKeyHash> metadata;
  std::unordered_map<ColumnKey, std::vector<ColumnRangePtr>, ColumnKeyHash>
      columns;
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
    : impl_(std::make_unique<Impl>()) {}

CudfDecodedColumnCache::~CudfDecodedColumnCache() = default;

CudfDecodedColumnCache& CudfDecodedColumnCache::instance() {
  // Intentionally process-lifetime: avoids CUDA pool destruction during static
  // teardown and implements the prototype's non-evicting lifetime.
  static auto* cache = new CudfDecodedColumnCache();
  return *cache;
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
    MetadataPtr metadata) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->metadata.try_emplace(std::move(key), std::move(metadata))
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
    CompressionMode compressionMode) {
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
  (isCompressed ? impl_->insertedCompressedRanges : impl_->insertedRawRanges)
      .fetch_add(1, std::memory_order_relaxed);
  if (compressionAttempted) {
    impl_->compressionAttempts.fetch_add(1, std::memory_order_relaxed);
    impl_->compressionEncodeNanos.fetch_add(
        compressionEncodeNanos, std::memory_order_relaxed);
  }
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

uint64_t CudfDecodedColumnCache::pinnedBytes() const {
  return impl_->allocatedBytes.load(std::memory_order_relaxed);
}

CudfDecodedColumnCache::Stats CudfDecodedColumnCache::stats() const {
  return {
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
  };
}

void CudfDecodedColumnCache::clearForTesting() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->columns.clear();
  impl_->metadata.clear();
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
}

} // namespace facebook::velox::cudf_velox::connector::hive
