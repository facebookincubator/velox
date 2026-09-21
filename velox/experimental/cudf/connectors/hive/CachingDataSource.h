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

/// Wraps a KvikIO cudf::io::datasource and serves its reads out of Velox's
/// AsyncDataCache.
///
/// Cache keys are (file ID, request offset). A read consumes the cached entry
/// at its offset even when that entry is shorter than the request, then
/// continues at the next offset and fills only the missing suffix. This is
/// exact-offset fragment reuse, not an arbitrary interval lookup: a read that
/// begins inside an existing entry does not reuse it.
///
/// Misses are filled with the delegate's asynchronous host read directly into
/// the cache entry. Pending fills and waits on exclusive entries do not occupy
/// executor threads, and the number of admitted logical reads is bounded before
/// any cache entry is allocated. Short or failed fills are never published.
///
/// Device reads stage the bytes through a per-thread pinned buffer. The future
/// returned by device_read_async fences the H2D copy both when it is consumed
/// and when it is discarded. Asynchronous work owns the delegate, so this
/// object may be destroyed while reads are in flight. The process-wide cache
/// must outlive all reads.
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

/// Returns 'delegate' itself when caching does not apply, that is when no
/// cache is configured or the split is not cacheable. The direct KvikIO path
/// then keeps its native asynchronous interfaces without an extra wrapper,
/// pool, file ID or staging allocation.
std::unique_ptr<cudf::io::datasource> maybeCacheKvikioDataSource(
    std::unique_ptr<cudf::io::datasource> delegate,
    std::string_view path,
    cache::AsyncDataCache* cache,
    bool cacheable,
    std::shared_ptr<IoStats> ioStats = nullptr);

} // namespace facebook::velox::cudf_velox::connector::hive
