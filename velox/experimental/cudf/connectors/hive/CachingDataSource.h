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

/// Serves reads of a KvikIO datasource through the AsyncDataCache. A miss reads
/// the bytes from the delegate into a new cache entry. Device reads copy each
/// cached piece to the GPU through a reusable per-thread pinned buffer. This is
/// separate from the BufferedInput/AWS SDK datasource and does not change that
/// reader's I/O.
/// device_read_async() fills the cache on one pool and copies to the GPU on a
/// second, smaller pool of cudf.kvikio_cache_threads threads. Its future also
/// waits for the GPU copy, whether it is consumed or discarded. Failed or
/// short fills are never published.
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
  std::unique_ptr<datasource::buffer>
  device_read(size_t offset, size_t size, cuda::stream_ref stream) override;

 private:
  struct State;
  // Asynchronous work owns the delegate and file ID independently of the
  // datasource object's lifetime.
  std::shared_ptr<State> state_;
};

/// With caching disabled (or a non-cacheable split), return the original
/// datasource itself. This preserves the direct KvikIO path, including native
/// async interfaces, without extra pools, file IDs or staging allocations.
/// Presto marks splits cacheable only under soft or hard affinity scheduling,
/// so with its default NO_PREFERENCE the cache is never used.
std::unique_ptr<cudf::io::datasource> maybeCacheKvikioDataSource(
    std::unique_ptr<cudf::io::datasource> delegate,
    std::string_view path,
    cache::AsyncDataCache* cache,
    bool cacheable,
    std::shared_ptr<IoStats> ioStats = nullptr);

} // namespace facebook::velox::cudf_velox::connector::hive
