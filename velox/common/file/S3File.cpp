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

#include "velox/common/file/S3File.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <fcntl.h>
#include <folly/portability/SysUio.h>
#include <sys/types.h>

namespace facebook::velox {

inline Aws::String getAwsString(const std::string& s) {
  return Aws::String(s.begin(), s.end());
}

inline std::string_view fromAwsString(const Aws::String& s) {
  return {s.data(), s.length()};
}

void S3FileSystem::configureDefaultCredentialChain() {
  credentials_provider_ =
      std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
}
void S3FileSystem::configureAccessKey() {
  credentials_provider_ =
      std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>(
          getAwsString((config_->get("hive.s3.aws-access-key")).value()),
          getAwsString((config_->get("hive.s3.aws-secret-key")).value()),
          getAwsString(""));
}

std::string S3FileSystem::getAccessKey() const {
  auto credentials = credentials_provider_->GetAWSCredentials();
  return std::string(fromAwsString(credentials.GetAWSAccessKeyId()));
}

std::string S3FileSystem::getSecretKey() const {
  auto credentials = credentials_provider_->GetAWSCredentials();
  return std::string(fromAwsString(credentials.GetAWSSecretKey()));
}

std::string S3FileSystem::getSessionToken() const {
  auto credentials = credentials_provider_->GetAWSCredentials();
  return std::string(fromAwsString(credentials.GetSessionToken()));
}

std::mutex aws_init_lock;
Aws::SDKOptions aws_options;
std::atomic<bool> aws_initialized(false);

void InitializeS3() {
  std::lock_guard<std::mutex> lock(aws_init_lock);
  if (!aws_initialized.load()) {
    aws_options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Fatal;
    Aws::InitAPI(aws_options);
    aws_initialized.store(true);
  }
}

void S3FileSystem::init() {
  if (!getRegion().empty()) {
    client_config_.region = getAwsString(getRegion());
  }
  client_config_.endpointOverride = getEndPoint();
  if (getScheme() == "http") {
    client_config_.scheme = Aws::Http::Scheme::HTTP;
  } else if (getScheme() == "https") {
    client_config_.scheme = Aws::Http::Scheme::HTTPS;
  } else {
    throw std::runtime_error(
        fmt::format("Invalid S3 connection scheme '", getScheme(), "'"));
  }

  // use virtual addressing for S3 on AWS (end point is empty)
  const bool use_virtual_addressing = getEndPoint().empty();
  configureAccessKey();
  client_ = std::make_shared<Aws::S3::S3Client>(
      credentials_provider_,
      client_config_,
      Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
      use_virtual_addressing);
}

void S3ReadFile::preadInternal(uint64_t offset, uint64_t length, char* pos)
    const {
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

std::string_view
S3ReadFile::pread(uint64_t offset, uint64_t length, Arena* arena) const {
  char* pos = arena->reserve(length);
  preadInternal(offset, length, pos);
  return {pos, length};
}

std::string_view S3ReadFile::pread(uint64_t offset, uint64_t length, void* buf)
    const {
  preadInternal(offset, length, static_cast<char*>(buf));
  return {static_cast<char*>(buf), length};
}

std::string S3ReadFile::pread(uint64_t offset, uint64_t length) const {
  // TODO: use allocator that doesn't initialize memory?
  std::string result(length, 0);
  char* pos = result.data();
  preadInternal(offset, length, pos);
  return result;
}

uint64_t S3ReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) {
  throw std::runtime_error("failure in S3ReadFile::preadv.");
  return -1;
}

uint64_t S3ReadFile::size() const {
  throw std::runtime_error("failure in S3ReadFile::size.");
}

uint64_t S3ReadFile::memoryUsage() const {
  throw std::runtime_error("failure in S3ReadFile::memoryUsage.");
  return 0;
}
// Register S3 Files.
namespace {

struct S3FileRegistrar {
  S3FileRegistrar() {
    lazyRegisterFileClass([this]() { this->ActuallyRegister(); });
  }

  void ActuallyRegister() {
    // Note: presto behavior is to prefix local paths with 'file:'.
    // Check for that prefix and prune to absolute regular paths as needed.
    std::function<bool(std::string_view)> filename_matcher =
        [](std::string_view filename) { return isS3File(filename); };
    std::function<std::unique_ptr<ReadFile>(
        std::string_view, std::shared_ptr<const Config>)>
        read_generator = [](std::string_view filename,
                            std::shared_ptr<const Config> properties) {
          // TODO: Cache the FileSystem
          auto s3fs = S3FileSystem(properties);
          s3fs.init();
          return s3fs.openReadFile(filename);
        };
    std::function<std::unique_ptr<WriteFile>(
        std::string_view, std::shared_ptr<const Config>)>
        write_generator = [](std::string_view filename,
                             std::shared_ptr<const Config> properties) {
          // TODO: Cache the FileSystem
          return S3FileSystem(properties).openWriteFile(filename);
        };
    registerFileClass(filename_matcher, read_generator, write_generator);
  }
};

const S3FileRegistrar s3FileRegistrar;

} // namespace

} // namespace facebook::velox
