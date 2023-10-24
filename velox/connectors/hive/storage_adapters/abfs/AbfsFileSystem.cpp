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
#include "velox/connectors/hive/storage_adapters/abfs/AbfsFileSystem.h"
#include "velox/common/file/File.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsReadFile.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsWriteFile.h"
#include "velox/core/Config.h"

#include <azure/storage/blobs/blob_client.hpp>
#include <azure/storage/files/datalake.hpp>
#include <fmt/format.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <glog/logging.h>

namespace facebook::velox::filesystems::abfs {
using namespace Azure::Storage::Blobs;

class AbfsConfig {
 public:
  AbfsConfig(const Config* config) : config_(config) {}

  std::string connectionString(const std::string& path) const {
    auto abfsAccount = AbfsAccount(path);
    auto key = abfsAccount.credKey();
    VELOX_USER_CHECK(
        config_->isValueExists(key), "Failed to find storage credentials");

    return abfsAccount.connectionString(config_->get(key).value());
  }

 private:
  const Config* FOLLY_NONNULL config_;
};

class AbfsReadFile::Impl {
  constexpr static uint64_t kNaturalReadSize = 4 << 20; // 4M
  constexpr static uint64_t kReadConcurrency = 8;

 public:
  explicit Impl(const std::string& path, const std::string& connectStr)
      : path_(path), connectStr_(connectStr) {
    auto abfsAccount = AbfsAccount(path_);
    fileSystem_ = abfsAccount.fileSystem();
    fileName_ = abfsAccount.filePath();
    fileClient_ =
        std::make_unique<BlobClient>(BlobClient::CreateFromConnectionString(
            connectStr_, fileSystem_, fileName_));
  }

  void initialize() {
    if (length_ != -1) {
      return;
    }
    try {
      auto properties = fileClient_->GetProperties();
      length_ = properties.Value.BlobSize;
    } catch (Azure::Storage::StorageException& e) {
      throwStorageExceptionWithOperationDetails("GetProperties", fileName_, e);
    }

    VELOX_CHECK_GE(length_, 0);
  }

  std::string_view pread(uint64_t offset, uint64_t length, void* buffer) const {
    preadInternal(offset, length, static_cast<char*>(buffer));
    return {static_cast<char*>(buffer), length};
  }

  std::string pread(uint64_t offset, uint64_t length) const {
    std::string result(length, 0);
    preadInternal(offset, length, result.data());
    return result;
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers) const {
    size_t length = 0;
    auto size = buffers.size();
    for (auto& range : buffers) {
      length += range.size();
    }
    std::string result(length, 0);
    preadInternal(offset, length, static_cast<char*>(result.data()));
    size_t resultOffset = 0;
    for (auto range : buffers) {
      if (range.data()) {
        memcpy(range.data(), &(result.data()[resultOffset]), range.size());
      }
      resultOffset += range.size();
    }

    return length;
  }

  void preadv(
      folly::Range<const common::Region*> regions,
      folly::Range<folly::IOBuf*> iobufs) const {
    VELOX_CHECK_EQ(regions.size(), iobufs.size());
    for (size_t i = 0; i < regions.size(); ++i) {
      const auto& region = regions[i];
      auto& output = iobufs[i];
      output = folly::IOBuf(folly::IOBuf::CREATE, region.length);
      pread(region.offset, region.length, output.writableData());
      output.append(region.length);
    }
  }

  uint64_t size() const {
    return length_;
  }

  uint64_t memoryUsage() const {
    return 3 * sizeof(std::string) + sizeof(int64_t);
  }

  bool shouldCoalesce() const {
    return false;
  }

  std::string getName() const {
    return fileName_;
  }

  uint64_t getNaturalReadSize() const {
    return kNaturalReadSize;
  }

 private:
  void preadInternal(uint64_t offset, uint64_t length, char* position) const {
    // Read the desired range of bytes.
    Azure::Core::Http::HttpRange range;
    range.Offset = offset;
    range.Length = length;

    Azure::Storage::Blobs::DownloadBlobOptions blob;
    blob.Range = range;

    auto response = fileClient_->Download(blob);
    response.Value.BodyStream->ReadToCount(
        reinterpret_cast<uint8_t*>(position), length);
  }

  const std::string path_;
  const std::string connectStr_;
  std::string fileSystem_;
  std::string fileName_;
  std::unique_ptr<BlobClient> fileClient_;

  int64_t length_ = -1;
};

AbfsReadFile::AbfsReadFile(
    const std::string& path,
    const std::string& connectStr) {
  impl_ = std::make_shared<Impl>(path, connectStr);
}

void AbfsReadFile::initialize() {
  return impl_->initialize();
}

std::string_view
AbfsReadFile::pread(uint64_t offset, uint64_t length, void* buffer) const {
  return impl_->pread(offset, length, buffer);
}

std::string AbfsReadFile::pread(uint64_t offset, uint64_t length) const {
  return impl_->pread(offset, length);
}

uint64_t AbfsReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) const {
  return impl_->preadv(offset, buffers);
}

void AbfsReadFile::preadv(
    folly::Range<const common::Region*> regions,
    folly::Range<folly::IOBuf*> iobufs) const {
  return impl_->preadv(regions, iobufs);
}

uint64_t AbfsReadFile::size() const {
  return impl_->size();
}

uint64_t AbfsReadFile::memoryUsage() const {
  return impl_->memoryUsage();
}

bool AbfsReadFile::shouldCoalesce() const {
  return false;
}

std::string AbfsReadFile::getName() const {
  return impl_->getName();
}

uint64_t AbfsReadFile::getNaturalReadSize() const {
  return impl_->getNaturalReadSize();
}

class BlobStorageFileClient final : public IBlobStorageFileClient {
 public:
  BlobStorageFileClient(const DataLakeFileClient client)
      : client_(std::make_unique<DataLakeFileClient>(client)) {}

  void Create() override {
    client_->Create();
  }

  PathProperties GetProperties() override {
    return client_->GetProperties().Value;
  }

  void Append(const uint8_t* buffer, size_t size, uint64_t offset) override {
    auto bodyStream = Azure::Core::IO::MemoryBodyStream(buffer, size);
    client_->Append(bodyStream, offset);
  }

  void Flush(uint64_t position) override {
    client_->Flush(position);
  }

  void Close() override {
    // do nothing.
  }

 private:
  std::unique_ptr<DataLakeFileClient> client_;
};

class AbfsWriteFile::Impl {
 public:
  explicit Impl(const std::string& path, const std::string& connectStr)
      : path_(path), connectStr_(connectStr) {
    // Make it a no-op if invoked twice.
    if (position_ != -1) {
      return;
    }
    position_ = 0;
  }

  void initialize() {
    if (!blobStorageFileClient_) {
      auto abfsAccount = AbfsAccount(path_);
      auto fileClient = DataLakeFileClient::CreateFromConnectionString(
          connectStr_, abfsAccount.fileSystem(), abfsAccount.filePath());
      blobStorageFileClient_ = std::make_unique<BlobStorageFileClient>(
          BlobStorageFileClient(fileClient));
    }

    VELOX_CHECK(!exist(), "File already exists");
    blobStorageFileClient_->Create();
  }

  /// mainly for test purpose.
  void setFileClient(
      std::shared_ptr<IBlobStorageFileClient> blobStorageManager) {
    blobStorageFileClient_ = std::move(blobStorageManager);
  }

  bool exist() {
    try {
      blobStorageFileClient_->GetProperties();
      return true;
    } catch (Azure::Storage::StorageException& e) {
      if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
        return false;
      } else {
        throwStorageExceptionWithOperationDetails("GetProperties", path_, e);
      }
    }
  }

  void close() {
    if (!closed_) {
      flush();
      blobStorageFileClient_->Close();
      closed_ = true;
    }
  }

  void flush() {
    if (!closed_) {
      blobStorageFileClient_->Flush(position_);
    }
  }

  void append(std::string_view data) {
    VELOX_CHECK(!closed_, "File is not open");
    if (data.size() == 0) {
      return;
    }
    append(data.data(), data.size());
  }

  uint64_t size() const {
    auto properties = blobStorageFileClient_->GetProperties();
    return properties.FileSize;
  }

  void append(const char* buffer, size_t size) {
    auto offset = position_;
    position_ += size;
    blobStorageFileClient_->Append(
        reinterpret_cast<const uint8_t*>(buffer), size, offset);
  }

 private:
  const std::string path_;
  const std::string connectStr_;
  std::string fileSystem_;
  std::string fileName_;
  std::shared_ptr<IBlobStorageFileClient> blobStorageFileClient_;

  uint64_t position_ = -1;
  std::atomic<bool> closed_{false};
};

AbfsWriteFile::AbfsWriteFile(
    const std::string& path,
    const std::string& connectStr) {
  impl_ = std::make_shared<Impl>(path, connectStr);
}

void AbfsWriteFile::initialize() {
  impl_->initialize();
}

void AbfsWriteFile::close() {
  impl_->close();
}

void AbfsWriteFile::flush() {
  impl_->flush();
}

void AbfsWriteFile::append(std::string_view data) {
  impl_->append(data);
}

uint64_t AbfsWriteFile::size() const {
  return impl_->size();
}

void AbfsWriteFile::setFileClient(
    std::shared_ptr<IBlobStorageFileClient> fileClient) {
  impl_->setFileClient(std::move(fileClient));
}

class AbfsFileSystem::Impl {
 public:
  explicit Impl(const Config* config) : abfsConfig_(config) {
    LOG(INFO) << "Init Azure Blob file system";
  }

  ~Impl() {
    LOG(INFO) << "Dispose Azure Blob file system";
  }

  const std::string connectionString(const std::string& path) const {
    // Extract account name
    return abfsConfig_.connectionString(path);
  }

 private:
  const AbfsConfig abfsConfig_;
  std::shared_ptr<folly::Executor> ioExecutor_;
};

AbfsFileSystem::AbfsFileSystem(const std::shared_ptr<const Config>& config)
    : FileSystem(config) {
  impl_ = std::make_shared<Impl>(config.get());
}

std::string AbfsFileSystem::name() const {
  return "ABFS";
}

std::unique_ptr<ReadFile> AbfsFileSystem::openFileForRead(
    std::string_view path,
    const FileOptions& /*unused*/) {
  auto abfsfile = std::make_unique<AbfsReadFile>(
      std::string(path), impl_->connectionString(std::string(path)));
  abfsfile->initialize();
  return abfsfile;
}

std::unique_ptr<WriteFile> AbfsFileSystem::openFileForWrite(
    std::string_view path,
    const FileOptions& /*unused*/) {
  auto abfsfile = std::make_unique<AbfsWriteFile>(
      std::string(path), impl_->connectionString(std::string(path)));
  abfsfile->initialize();
  return abfsfile;
}
} // namespace facebook::velox::filesystems::abfs
