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

#include <google/cloud/storage/client.h>


namespace facebook::velox {
namespace filesystems {
////////////////////
namespace {
namespace gcs = ::google::cloud::storage;

// Reference: https://issues.apache.org/jira/browse/ARROW-14346
// https://github.com/apache/arrow/blob/master/cpp/src/arrow/filesystem/gcsfs.cc#L49
// Change the default upload buffer size. In general, sending larger buffers is more
// efficient with GCS, as each buffer requires a roundtrip to the service. With formatted
// output (when using `operator<<`), keeping a larger buffer in memory before uploading
// makes sense.  With unformatted output (the only choice given gcs::io::OutputStream's
// API) it is better to let the caller provide as large a buffer as they want. The GCS C++
// client library will upload this buffer with zero copies if possible.
auto constexpr kUploadBufferSize = 256 * 1024;


class GCSReadFile final : public ReadFile {
 public:
  GCSReadFile(const std::string& path, gcs::Client* client)
      : client_(client) {
    //assumption it's a proper path
    bucketAndKeyFromGCSPath(path, bucket_, key_);
  }

  // Gets the length of the file.
  // Checks if there are any issues reading the file.
  void initialize() {
    // Make it a no-op if invoked twice.
    if (length_ != -1) {
      return;
    }
    //get metadata and initialize length
    auto metadata = client_->GetObjectMetadata(bucket_, key_);
    VELOX_CHECK_GCS_OUTCOME(
        metadata.status(), "Failed to get metadata for GCS object", bucket_, key_);
    length_ =  (*metadata).size();
    VELOX_CHECK_GE(length_, 0);
  }

  std::string_view pread(uint64_t offset, uint64_t length, void* buffer)
      const override {
    preadInternal(offset, length, static_cast<char*>(buffer));
    return {static_cast<char*>(buffer), length};
  }

  std::string pread(uint64_t offset, uint64_t length) const override {
    std::string result(length, 0);
    char* position = result.data();
    preadInternal(offset, length, position);
    return result;
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers) const override {
    // 'buffers' contains Ranges(data, size)  with some gaps (data = nullptr) in
    // between. This call must populate the ranges (except gap ranges)
    // sequentially starting from 'offset'. If a range pointer is nullptr, the data
    // from stream of size range.size() will be skipped.
    size_t length = 0;
    for (const auto range : buffers) {
      length += range.size();
    }
    // TODO: allocate from a memory pool
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

  uint64_t size() const override {
    return length_;
  }

  uint64_t memoryUsage() const override {
    // TODO: Check if any buffers are being used by the GCS library
    return sizeof(gcs::Client) + kUploadBufferSize + 2 * sizeof(std::string) +
        sizeof(int64_t);
  }

  bool shouldCoalesce() const final {
    return false;
  }

 private:
  // The assumption here is that "position" has space for at least "length"
  // bytes.
  void preadInternal(uint64_t offset, uint64_t length, char* position) const {
    gcs::ObjectReadStream stream = client_->ReadObject(bucket_, key_,  gcs::ReadRange(offset, offset+length));
    VELOX_CHECK_GCS_OUTCOME(stream.status(), "Failed to get GCS object", bucket_, key_);

    stream.read(position, length);
    VELOX_CHECK_GCS_OUTCOME(stream.status(), "Failed to get read object", bucket_, key_);
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
  GCSConfig(const Config* config) : config_(config)
  {
    std::string cred = config_->get("credentials", std::string(""));
    if (!cred.empty())
    {
      credentials_ = google::cloud::MakeServiceAccountCredentials(cred);
    }
  }

  std::string endpointOverride() const {
    return config_->get("endpoint_override", std::string(""));
  }

  std::string scheme() const {

    return config_->get("scheme", std::string("https"));
  }

  std::shared_ptr<google::cloud::Credentials> credentials() const
  {
   return credentials_;
  }
 private:
  const Config* FOLLY_NONNULL config_;
  std::shared_ptr<google::cloud::Credentials> credentials_;
};

class GCSFileSystem::Impl {
 public:
  Impl(const Config* config) : gcsConfig_(config)
  {
  }

  ~Impl() {

  }

  // Use the input Config parameters and initialize the GCSClient.
  void initializeClient() {

    auto options = google::cloud::Options{};
    std::string scheme = gcsConfig_.scheme();
    //if (scheme.empty()) scheme = "https";
    if (scheme == "https") {
      options.set<google::cloud::UnifiedCredentialsOption>(
          google::cloud::MakeGoogleDefaultCredentials());
    } else {
      options.set<google::cloud::UnifiedCredentialsOption>(
          google::cloud::MakeInsecureCredentials());
    }
    options.set<gcs::UploadBufferSizeOption>(kUploadBufferSize);
    if (!gcsConfig_.endpointOverride().empty()) {
      options.set<gcs::RestEndpointOption>(scheme + "://" + gcsConfig_.endpointOverride());
    }
    if (gcsConfig_.credentials()) {
      options.set<google::cloud::UnifiedCredentialsOption>(gcsConfig_.credentials());
    }
    client_ = std::make_shared<gcs::Client>(options);
  }

  gcs::Client* gcsClient() const {
    return client_.get();
  }

 private:
  const GCSConfig gcsConfig_;
  std::shared_ptr<gcs::Client> client_;
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
  auto gcsfile = std::make_unique<GCSReadFile>(file, impl_->gcsClient());
  gcsfile->initialize();
  return gcsfile;
}

void GCSFileSystem::remove(std::string_view path)
{
  if (!isGCSFile(path))
    VELOX_FAIL("File {} is not a valid gcs file", path);
  const std::string file = gcsPath(path);

  //assumption it's a proper path
  std::string bucket;
  std::string object;
  bucketAndKeyFromGCSPath(file, bucket, object);

  if (!object.empty()) {
    auto stat = impl_->gcsClient()->GetObjectMetadata(bucket, object);
    VELOX_CHECK_GCS_OUTCOME(
      stat.status(), "Failed to get metadata for GCS object", bucket, object);
  }
  auto ret = impl_->gcsClient()->DeleteObject(bucket, object);
  VELOX_CHECK_GCS_OUTCOME(
      ret, "Failed to get metadata for GCS object", bucket, object);
}
std::string GCSFileSystem::name() const {
  return "GCS";
}

folly::once_flag GCSInstantiationFlag;
static std::function<std::shared_ptr<FileSystem>(std::shared_ptr<const Config>)>
    filesystemGenerator = [](std::shared_ptr<const Config> properties) {
      // Only one instance of GCSFileSystem is supported for now (follow S3 for now).
      // TODO: Support multiple GCSFileSystem instances using a cache
      // Initialize on first access and reuse after that.
      static std::shared_ptr<FileSystem> gcsfs;
      folly::call_once(GCSInstantiationFlag, [&properties]() {
        std::shared_ptr<GCSFileSystem> fs;
        if (properties != nullptr) {
          fs = std::make_shared<GCSFileSystem>(properties);
        } else {
          fs = std::make_shared<GCSFileSystem>(
              std::make_shared<core::MemConfig>());
        }
        fs->initializeClient();
        gcsfs = fs;
      });
      return gcsfs;
    };

void registerGCSFileSystem() {
  registerFileSystem(isGCSFile, filesystemGenerator);
}

} // namespace filesystems
} // namespace facebook::velox
