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
#include "velox/connectors/hive/storage_adapters/abfs/AbfsConfig.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"

namespace facebook::velox::filesystems {
class DataLakeFileClientWrapper final : public AzureDataLakeFileClient {
 public:
  DataLakeFileClientWrapper(std::unique_ptr<DataLakeFileClient> client)
      : client_(std::move(client)) {}

  void create() override {
    client_->Create();
  }

  Azure::Storage::Files::DataLake::Models::PathProperties getProperties()
      override {
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
  const std::unique_ptr<DataLakeFileClient> client_;
};

class AbfsWriteFile::Impl {
 public:
  explicit Impl(
      std::string_view path,
      std::unique_ptr<AzureDataLakeFileClient>& client)
      : path_(path), client_(std::move(client)) {
    // Make it a no-op if invoked twice.
    if (position_ != -1) {
      return;
    }
    position_ = 0;
    VELOX_CHECK(!checkIfFileExists(), "File already exists");
    client_->create();
  }

  void close() {
    if (!closed_) {
      flush();
      closed_ = true;
    }
  }

  void flush() {
    if (!closed_) {
      client_->flush(position_);
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
    return client_->getProperties().FileSize;
  }

  void append(const char* buffer, size_t size) {
    client_->append(reinterpret_cast<const uint8_t*>(buffer), size, position_);
    position_ += size;
  }

 private:
  bool checkIfFileExists() {
    try {
      client_->getProperties();
      return true;
    } catch (Azure::Storage::StorageException& e) {
      if (e.StatusCode != Azure::Core::Http::HttpStatusCode::NotFound) {
        throwStorageExceptionWithOperationDetails("GetProperties", path_, e);
      }
      return false;
    }
  }

  const std::string path_;
  const std::unique_ptr<AzureDataLakeFileClient> client_;

  uint64_t position_ = -1;
  bool closed_ = false;
};

AbfsWriteFile::AbfsWriteFile(
    std::string_view path,
    const config::ConfigBase& config) {
  auto abfsConfig = AbfsConfig(path, config);
  std::unique_ptr<AzureDataLakeFileClient> clientWrapper =
      std::make_unique<DataLakeFileClientWrapper>(
          abfsConfig.getWriteFileClient());
  impl_ = std::make_unique<Impl>(path, clientWrapper);
}

AbfsWriteFile::AbfsWriteFile(
    std::string_view path,
    std::unique_ptr<AzureDataLakeFileClient>& client) {
  impl_ = std::make_unique<Impl>(path, client);
}

AbfsWriteFile::~AbfsWriteFile() {}

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

} // namespace facebook::velox::filesystems
