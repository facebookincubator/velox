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

#include "velox/adapters/s3fs/S3FileSystem.h"
#include "velox/common/file/File.h"
#include "velox/core/Context.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <aws/core/Aws.h>
#include <aws/core/Region.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>

namespace facebook::velox {

constexpr std::string_view kSep{"/"};
constexpr std::string_view s3Scheme{"s3:"};

bool inline isS3File(const std::string_view filename) {
  return (filename.substr(0, s3Scheme.size()) == s3Scheme);
}

inline Aws::String getAwsString(const std::string& s) {
  return Aws::String(s.begin(), s.end());
}

inline std::string_view fromAwsString(const Aws::String& s) {
  return {s.data(), s.length()};
}

// Implement the S3ReadFile
class S3ReadFile final : public ReadFile {
 public:
  S3ReadFile(std::string path, std::shared_ptr<Aws::S3::S3Client> client)
      : path_(std::move(path)), client_(client) {}

  std::string_view pread(uint64_t offset, uint64_t length, Arena* arena)
      const override {
    char* pos = arena->reserve(length);
    preadInternal(offset, length, pos);
    return {pos, length};
  }

  std::string_view pread(uint64_t offset, uint64_t length, void* buf)
      const override {
    preadInternal(offset, length, static_cast<char*>(buf));
    return {static_cast<char*>(buf), length};
  }

  std::string pread(uint64_t offset, uint64_t length) const override {
    // TODO: use allocator that doesn't initialize memory?
    std::string result(length, 0);
    char* pos = result.data();
    preadInternal(offset, length, pos);
    return result;
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers) override {
    throw std::runtime_error("failure in S3ReadFile::preadv.");
    return -1;
  }

  uint64_t size() const override {
    throw std::runtime_error("failure in S3ReadFile::size.");
  }

  uint64_t memoryUsage() const override {
    throw std::runtime_error("failure in S3ReadFile::memoryUsage.");
    return 0;
  }

  bool shouldCoalesce() const final {
    return false;
  }

 private:
  void preadInternal(uint64_t offset, uint64_t length, char* pos) const {
    // Read the desired range of bytes
    Aws::S3::Model::GetObjectRequest req;
    Aws::S3::Model::GetObjectResult result;
    auto first_sep = path_.find_first_of(kSep);
    req.SetBucket(getAwsString(path_.substr(0, first_sep)));
    req.SetKey(getAwsString(path_.substr(first_sep + 1)));
    std::stringstream ss;
    ss << "bytes=" << offset << "-" << offset + length - 1;
    req.SetRange(getAwsString(ss.str()));
    // TODO: Avoid copy below by using  req.SetResponseStreamFactory();
    // Reference: ARROW-8692
    auto outcome = client_->GetObject(req);
    if (!outcome.IsSuccess()) {
      throw std::runtime_error("failure in S3ReadFile::size preadInternal.");
    }
    result = std::move(outcome).GetResultWithOwnership();
    auto& stream = result.GetBody();
    stream.read(reinterpret_cast<char*>(pos), length);
  }
  std::shared_ptr<Aws::S3::S3Client> client_;
  std::string path_;
};

namespace filesystems {
std::once_flag s3InitializeFlag;

void initializeS3Library() {
  std::call_once(s3InitializeFlag, []() {
    Aws::SDKOptions aws_options;
    aws_options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Fatal;
    Aws::InitAPI(aws_options);
  });
}

namespace S3Config {
constexpr char const* pathAccessStyle{"hive.s3.path-style-access"};
constexpr char const* endpoint{"hive.s3.endpoint"};
constexpr char const* secretKey{"hive.s3.aws-secret-key"};
constexpr char const* accessKey{"hive.s3.aws-access-key"};
constexpr char const* sslEnabled{"hive.s3.ssl.enabled"};
constexpr char const* useInstanceCredentials{
    "hive.s3.use-instance-credentials"};
} // namespace S3Config

// Implement the S3FileSystem
class S3FileSystem::Impl {
 public:
  Impl(const Config* config) : config_(config) {}

  std::string getOptionalProperty(
      const std::string& name,
      const std::string& defaultValue) {
    auto value = config_->get(name);
    if (!value.hasValue()) {
      return defaultValue;
    }
    return value.value();
  }

  // Configure default AWS credentials provider chain.
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider>
  getDefaultCredentialProvider() {
    return std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
  }

  // Configure with access and secret keys. Used for on-prem.
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider>
  getAccessSecretCredentialProvider(
      std::string accessKey,
      std::string secretKey) {
    return std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>(
        getAwsString(accessKey), getAwsString(secretKey), getAwsString(""));
  }

  // Use the input Config parameters and initialize the S3Client
  void initializeClient() {
    Aws::Client::ClientConfiguration clientConfig;

    const auto endpoint = getOptionalProperty(S3Config::endpoint, "");
    clientConfig.endpointOverride = endpoint;

    // default is use SSL
    const auto useSSL =
        (getOptionalProperty(S3Config::sslEnabled, "true") == "true");
    if (useSSL) {
      clientConfig.scheme = Aws::Http::Scheme::HTTPS;
    } else {
      clientConfig.scheme = Aws::Http::Scheme::HTTP;
    }

    // use virtual addressing for S3 on AWS
    const auto pathAccessStyle =
        getOptionalProperty(S3Config::pathAccessStyle, "false");
    const bool useVirtualAddressing = (pathAccessStyle == "false");

    const auto accessKey = getOptionalProperty(S3Config::accessKey, "");
    const auto secretKey = getOptionalProperty(S3Config::secretKey, "");
    const auto useInstanceCred =
        getOptionalProperty(S3Config::useInstanceCredentials, "");
    std::shared_ptr<Aws::Auth::AWSCredentialsProvider> credentials_provider;
    if (accessKey != "" && secretKey != "" && useInstanceCred != "true") {
      credentials_provider =
          getAccessSecretCredentialProvider(accessKey, secretKey);
    } else {
      credentials_provider = getDefaultCredentialProvider();
    }

    client_ = std::make_shared<Aws::S3::S3Client>(
        credentials_provider,
        clientConfig,
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
        useVirtualAddressing);
  }

  std::shared_ptr<Aws::S3::S3Client> getS3Client() {
    return client_;
  }

 private:
  const Config* config_;
  std::shared_ptr<Aws::S3::S3Client> client_;
};

S3FileSystem::S3FileSystem(std::shared_ptr<const Config> config)
    : FileSystem(config) {
  impl_ = std::make_shared<Impl>(config.get());
}

void S3FileSystem::initializeClient() {
  impl_->initializeClient();
}

std::unique_ptr<ReadFile> S3FileSystem::openFileForRead(std::string_view path) {
  // Remove the S3://
  const std::string bucket =
      std::string(path.substr(s3Scheme.length() + 2 * kSep.length()));
  return std::make_unique<S3ReadFile>(bucket, impl_->getS3Client());
}

std::unique_ptr<WriteFile> S3FileSystem::openFileForWrite(
    std::string_view path) {
  // Not yet implemented
  return nullptr;
}

std::string S3FileSystem::name() const {
  return "S3";
}

std::function<bool(std::string_view)> schemeMatcher =
    [](std::string_view filename) { return isS3File(filename); };

std::function<std::shared_ptr<FileSystem>(std::shared_ptr<const Config>)>
    filesystemGenerator = [](std::shared_ptr<const Config> properties) {
      initializeS3Library();
      // TODO: Cache the FileSystem
      auto s3fs = std::make_shared<S3FileSystem>(properties);
      s3fs->initializeClient();
      return s3fs;
    };

void registerS3FileSystem() {
  registerFileSystem(schemeMatcher, filesystemGenerator);
}

} // namespace filesystems
} // namespace facebook::velox
