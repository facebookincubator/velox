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

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/file/File.h"

#include <cudf/io/datasource.hpp>

#include <cuda/stream>

#include <future>
#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive {

/// KvikIO-backed cache fills, followed by a reusable per-thread pinned H2D
/// staging buffer. This is separate from the BufferedInput/AWS SDK datasource;
/// it does not change that reader's I/O.
/// Device-read cache misses use the delegate's asynchronous host API. Pending
/// fills and exclusive-entry waits do not occupy remote executor threads.
/// Admission is bounded before allocating cache entries, independently of
/// driver/preload counts. The returned future still fences destination writes
/// on both consumption and discard; failed/short fills are never published.
///
/// Cache keys are (file ID, request offset). Reads reuse smaller entries at
/// successive offsets and fill only the remaining suffix. This is exact-offset
/// fragment reuse, not an arbitrary interval lookup: differently aligned
/// ranges need not reuse cached bytes. The process-wide cache must outlive
/// reads.
class CachingDataSource final : public cudf::io::datasource {
 public:
  CachingDataSource(
      std::unique_ptr<cudf::io::datasource> delegate,
      std::string_view path,
      cache::AsyncDataCache* cache,
      std::shared_ptr<IoStats> ioStats = nullptr);
  ~CachingDataSource() override;

  size_t size() const override;
  bool supports_device_read() const override;
  bool is_device_read_preferred(size_t size) const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;
  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst)
      override;
  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream) override;
  size_t device_read(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream) override;
  std::unique_ptr<datasource::buffer> device_read(
      size_t offset,
      size_t size,
      cuda::stream_ref stream) override;

 private:
  struct State;
  // Asynchronous work owns the delegate and file ID independently of the
  // datasource object's lifetime.
  std::shared_ptr<State> state_;
};

/// With caching disabled (or a non-cacheable split), return the original
/// datasource itself. This preserves the direct KvikIO path, including native
/// async interfaces, without extra pools, file IDs or staging allocations.
std::unique_ptr<cudf::io::datasource> maybeCacheKvikioDataSource(
    std::unique_ptr<cudf::io::datasource> delegate,
    std::string_view path,
    cache::AsyncDataCache* cache,
    bool cacheable,
    std::shared_ptr<IoStats> ioStats = nullptr);

} // namespace facebook::velox::cudf_velox::connector::hive
