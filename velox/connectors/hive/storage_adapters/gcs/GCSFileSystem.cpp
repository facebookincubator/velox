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

#include "velox/connectors/hive/storage_adapters/gcs/GCSFileSystem.h"
#include "velox/common/file/File.h"
#include "velox/connectors/hive/storage_adapters/gcs/GCSUtil.h"
#include "velox/core/Context.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <memory>
#include <stdexcept>

//TODO replace includes
#include <google/cloud/storage/client.h>

//TODO create a skeleton of interface that needs to be compiled
//TODO add a flag in main Makefile

namespace facebook::velox {
namespace filesystems {
////////////////////
namespace {
namespace gcs = ::google::cloud::storage;

class GCSReadFile final : public ReadFile {
 public:
  GCSReadFile(const std::string& path, gcs::Client* client)
      : client_(client) {
    //bucketAndKeyFromS3Path(path, bucket_, key_);
  }

  // Gets the length of the file.
  // Checks if there are any issues reading the file.
  void initialize() {
    // Make it a no-op if invoked twice.
    if (length_ != -1) {
      return;
    }

    //TODO
  }

  std::string_view pread(uint64_t offset, uint64_t length, void* buffer)
      const override {
   // preadInternal(offset, length, static_cast<char*>(buffer)); //TODO
    return {static_cast<char*>(buffer), length};
  }

  std::string pread(uint64_t offset, uint64_t length) const override {
    std::string result(length, 0);
    //char* position = result.data();
    //preadInternal(offset, length, position);
    //TODO
    return result;
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers) const override {
    return 0; //TODO
  }

  uint64_t size() const override {
    return length_;
  }

  uint64_t memoryUsage() const override {
    return 0; //TODO
  }

  bool shouldCoalesce() const final {
    return false;
  }

 private:
  // The assumption here is that "position" has space for at least "length"
  // bytes.
  void preadInternal(uint64_t offset, uint64_t length, char* position) const {
  }

  gcs::Client* client_;
  std::string bucket_;
  std::string key_;
  int64_t length_ = -1;
};
} // namespace

///////////////////
class GCSConfig {
 public:
  GCSConfig(const Config* config) : config_(config) {}

  //TODO fill in based on what's needed
 private:
  const Config* FOLLY_NONNULL config_;
};

class GCSFileSystem::Impl {
 public:
  Impl(const Config* config) : gcsConfig_(config) {

    //TODO
  }

  ~Impl() {
    //TODO
  }

  // Use the input Config parameters and initialize the GCSClient.
  void initializeClient() {
    //TODO
  }



 private:
  const GCSConfig gcsConfig_;
  //std::shared_ptr<Aws::S3::GCSClient> client_;
  //static std::atomic<size_t> initCounter_;
};

GCSFileSystem::GCSFileSystem(std::shared_ptr<const Config> config)
    : FileSystem(config) {
  impl_ = std::make_shared<Impl>(config.get());
}

void GCSFileSystem::initializeClient() {
  impl_->initializeClient();
}

std::unique_ptr<ReadFile> GCSFileSystem::openFileForRead(std::string_view path) {
  const std::string file = gcsPath(path);
  //auto gcsfile = std::make_unique<GCSReadFile>(file, impl_->s3Client());
  //gcsfile->initialize();
  return nullptr;
}

std::string GCSFileSystem::name() const {
  return "GCS";
}

static std::function<std::shared_ptr<FileSystem>(std::shared_ptr<const Config>)>
    filesystemGenerator = [](std::shared_ptr<const Config> properties) {
      // Only one instance of GCSFileSystem is supported for now.
      // TODO: Support multiple GCSFileSystem instances using a cache
      // Initialize on first access and reuse after that.
      static std::shared_ptr<FileSystem> gcsfs;
      //TODO
      return gcsfs;
    };

void registerGCSFileSystem() {
  registerFileSystem(isGCSFile, filesystemGenerator);
}

} // namespace filesystems
} // namespace facebook::velox
