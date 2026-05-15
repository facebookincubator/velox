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

#include "velox/experimental/cudf/connectors/hive/storage_adapters/abfs/CudfAbfsDataSource.h"
#include "velox/experimental/cudf/connectors/hive/storage_adapters/abfs/CudfAbfsPinnedBufferPool.h"

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h"

#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <azure/storage/blobs/blob_options.hpp>
#include <azure/storage/common/storage_exception.hpp>

#include <algorithm>
#include <future>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::filesystems {

namespace velox_filesystems = ::facebook::velox::filesystems;

namespace {

// Default upper bound on simultaneously outstanding pinned-host bytes
// reserved for ABFS device reads. Sized so that a handful of concurrent
// row-group reads fit comfortably; overridable via the connector
// property `cudf.hive.abfs-pinned-buffer-bytes`.
constexpr size_t kDefaultPinnedPoolBytes = 64ull * 1024ull * 1024ull;

// Connector / session property names. Kept here to avoid a build-time
// dependency from `velox_cudf_abfs` back into the hive connector
// library.
constexpr const char* kPinnedBufferBytes = "cudf.hive.abfs-pinned-buffer-bytes";

size_t resolvePoolBudget(
    const std::shared_ptr<const config::ConfigBase>& properties) {
  if (properties == nullptr) {
    return kDefaultPinnedPoolBytes;
  }
  return properties->get<size_t>(kPinnedBufferBytes, kDefaultPinnedPoolBytes);
}

} // namespace

CudfAbfsDataSource::CudfAbfsDataSource(
    std::string_view path,
    std::shared_ptr<const config::ConfigBase> properties)
    : path_(path),
      properties_(std::move(properties)),
      abfsPath_(
          std::make_shared<velox_filesystems::AbfsPath>(
              std::string_view{path_})) {
  VELOX_CHECK_NOT_NULL(
      properties_, "CudfAbfsDataSource requires non-null connector properties");
}

CudfAbfsDataSource::~CudfAbfsDataSource() = default;

void CudfAbfsDataSource::ensureInitialized() const {
  std::call_once(initFlag_, [&]() {
    fileClient_ =
        velox_filesystems::AzureClientProviderFactories::getReadFileClient(
            abfsPath_, *properties_);
    try {
      const auto properties = fileClient_->getProperties();
      fileSize_ = static_cast<size_t>(properties.Value.BlobSize);
    } catch (Azure::Storage::StorageException& e) {
      velox_filesystems::throwStorageExceptionWithOperationDetails(
          "GetProperties", abfsPath_->filePath(), e);
    }
  });
}

size_t CudfAbfsDataSource::size() const {
  ensureInitialized();
  return fileSize_;
}

void CudfAbfsDataSource::rangedDownload(
    size_t offset,
    size_t length,
    uint8_t* dst) const {
  ensureInitialized();
  if (length == 0) {
    return;
  }

  Azure::Core::Http::HttpRange range;
  range.Offset = static_cast<int64_t>(offset);
  range.Length = static_cast<int64_t>(length);

  Azure::Storage::Blobs::DownloadBlobOptions options;
  options.Range = range;

  try {
    auto response = fileClient_->download(options);
    response.Value.BodyStream->ReadToCount(dst, length);
  } catch (Azure::Storage::StorageException& e) {
    velox_filesystems::throwStorageExceptionWithOperationDetails(
        "Download", abfsPath_->filePath(), e);
  }
}

std::unique_ptr<cudf::io::datasource::buffer> CudfAbfsDataSource::host_read(
    size_t offset,
    size_t size) {
  ensureInitialized();
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  rangedDownload(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t CudfAbfsDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  ensureInitialized();
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  rangedDownload(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
CudfAbfsDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::async, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t>
CudfAbfsDataSource::host_read_async(size_t offset, size_t size, uint8_t* dst) {
  return std::async(std::launch::async, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

bool CudfAbfsDataSource::supports_device_read() const {
  return true;
}

namespace {

// Stream-ordered host callback that releases the pinned buffer handle
// once the device copy has completed. `userData` owns a
// heap-allocated handle and is deleted after release.
void releasePinnedBufferCallback(void* userData) {
  auto* handle = static_cast<CudfAbfsPinnedBufferPool::Handle*>(userData);
  delete handle;
}

} // namespace

std::future<size_t> CudfAbfsDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  ensureInitialized();

  return std::async(
      std::launch::async, [this, offset, size, dst, stream]() -> size_t {
        if (offset >= fileSize_ or size == 0) {
          return 0;
        }
        const size_t readSize = std::min(size, fileSize_ - offset);

        auto& pool =
            CudfAbfsPinnedBufferPool::shared(resolvePoolBudget(properties_));
        auto handle = pool.acquire(readSize);

        rangedDownload(offset, readSize, handle.data());

        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dst,
            handle.data(),
            readSize,
            cudaMemcpyHostToDevice,
            stream.value()));

        // Hand ownership of the pinned buffer to the stream so it lives
        // until the copy actually completes. The host callback frees the
        // handle, which returns the buffer to the pool.
        auto* heapHandle =
            new CudfAbfsPinnedBufferPool::Handle(std::move(handle));
        try {
          CUDF_CUDA_TRY(cudaLaunchHostFunc(
              stream.value(), &releasePinnedBufferCallback, heapHandle));
        } catch (...) {
          // Releasing inline is safe because a failed launch implies the
          // copy was not enqueued (or the runtime is unusable); reclaim
          // synchronously rather than leak.
          delete heapHandle;
          throw;
        }

        return readSize;
      });
}

} // namespace facebook::velox::cudf_velox::filesystems
