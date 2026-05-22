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

#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureBlobClient.h"

#include <cudf/io/datasource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive::io_sources {

class CudfAbfsPinnedBufferPool;

/// `cudf::io::datasource` implementation that reads bytes from Azure
/// Blob / Data Lake Storage directly through the upstream `AbfsPath`
/// and `AzureClientProviderFactories` plumbing. Bypasses Velox's
/// `BufferedInput` host-side staging on the data path. Today
/// `device_read_async` downloads into a pinned host buffer and issues a
/// `cudaMemcpyAsync` to the destination; when kvikIO grows native ABFS
/// support, swap that implementation here.
class CudfAbfsIoSource : public cudf::io::datasource {
 public:
  /// Constructs an IO source for `path`. `properties` must contain the
  /// same Azure credentials / endpoint keys used by the upstream
  /// `AbfsReadFile`. Parses `path` eagerly via `AbfsPath` so invalid
  /// URIs throw at construction time; the Azure SDK client and the file
  /// size are resolved lazily on first use.
  CudfAbfsIoSource(
      std::string_view path,
      std::shared_ptr<const config::ConfigBase> properties);

  ~CudfAbfsIoSource() override;

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst);

  [[nodiscard]] bool supports_device_read() const override;

  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      rmm::cuda_stream_view stream) override;

 private:
  // Ensures the Azure SDK client and the file size are resolved.
  // Idempotent across threads.
  void ensureInitialized() const;

  // Issues a ranged GET into `dst`. `length` must already be clamped to
  // the remaining bytes in the blob.
  void rangedDownload(size_t offset, size_t length, uint8_t* dst) const;

  std::string path_;
  std::shared_ptr<const config::ConfigBase> properties_;
  std::shared_ptr<facebook::velox::filesystems::AbfsPath> abfsPath_;

  mutable std::once_flag initFlag_;
  mutable std::unique_ptr<facebook::velox::filesystems::AzureBlobClient>
      fileClient_;
  mutable size_t fileSize_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive::io_sources
