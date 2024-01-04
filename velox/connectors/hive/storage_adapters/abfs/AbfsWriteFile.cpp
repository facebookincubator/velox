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

#include "velox/connectors/hive/storage_adapters/abfs/AbfsWriteFile.h"
#include <azure/storage/files/datalake.hpp>

namespace facebook::velox::filesystems::abfs {
class BlobStorageFileClient final : public IBlobStorageFileClient {
 public:
  BlobStorageFileClient(const DataLakeFileClient client)
      : client_(std::make_unique<DataLakeFileClient>(client)) {}

  void create() override {
    client_->Create();
  }

  PathProperties getProperties() override {
    return client_->GetProperties().Value;
  }

  void append(const uint8_t* buffer, size_t size, uint64_t offset) override {
    auto bodyStream = Azure::Core::IO::MemoryBodyStream(buffer, size);
    client_->Append(bodyStream, offset);
  }

  void flush(uint64_t position) override {
    client_->Flush(position);
  }

  void close() override {
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
    blobStorageFileClient_->create();
  }

  /// mainly for test purpose.
  void setFileClient(
      std::shared_ptr<IBlobStorageFileClient> blobStorageManager) {
    blobStorageFileClient_ = std::move(blobStorageManager);
  }

  bool exist() {
    try {
      blobStorageFileClient_->getProperties();
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
      blobStorageFileClient_->close();
      closed_ = true;
    }
  }

  void flush() {
    if (!closed_) {
      blobStorageFileClient_->flush(position_);
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
    auto properties = blobStorageFileClient_->getProperties();
    return properties.FileSize;
  }

  void append(const char* buffer, size_t size) {
    auto offset = position_;
    position_ += size;
    blobStorageFileClient_->append(
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
} // namespace facebook::velox::filesystems::abfs