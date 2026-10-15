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

/// Reads a KvikIO datasource through Velox's AsyncDataCache.
///
/// Each read is split into pieces, one per cache entry. A cached piece is
/// copied from the cache. A missing piece is read from KvikIO into a new cache
/// entry first. A piece that cannot be cached (the cache is full, or the piece
/// is too big) is read from KvikIO directly, together with the rest of the
/// read. A partly read entry is never made readable.
///
/// GPU reads copy each piece through a pinned host buffer, one per thread.
/// device_read_async() fills the cache on one thread pool and copies to the
/// GPU on a second, smaller pool of cudf.kvikio_cache_threads threads. Its
/// future waits until the bytes are in GPU memory, whether cuDF calls get() on
/// it or drops it.
///
/// Cache entries are keyed by (file, offset). A read can use a shorter cached
/// entry at its offset, then continue with the entry right after it. Entries
/// at other offsets are not used, even if they overlap the read.
///
/// This does not change the BufferedInput/AWS SDK reader. The process-wide
/// cache must outlive all reads.
class CachingDataSource final : public cudf::io::datasource {
 public:
  CachingDataSource(
      std::unique_ptr<cudf::io::datasource> kvikioSource,
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
  struct Reader;
  // Shared with async work, which may outlive this object.
  std::shared_ptr<Reader> reader_;
};

/// Returns 'kvikioSource' wrapped in a CachingDataSource, or 'kvikioSource'
/// itself when 'cache' is null or the split is not cacheable. Presto marks
/// splits cacheable only under soft or hard affinity scheduling, so with its
/// default NO_PREFERENCE the cache is never used.
std::unique_ptr<cudf::io::datasource> maybeCacheKvikioDataSource(
    std::unique_ptr<cudf::io::datasource> kvikioSource,
    std::string_view path,
    cache::AsyncDataCache* cache,
    bool cacheable,
    std::shared_ptr<IoStats> ioStats = nullptr);

} // namespace facebook::velox::cudf_velox::connector::hive
