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

// Implementation of S3 filesystem and file interface.
// We provide a registration method for read and write files so the appropriate
// type of file can be constructed based on a filename. See the
// (register|generate)ReadFile and (register|generate)WriteFile functions.

#pragma once

#include <aws/core/Aws.h>
#include <aws/core/Region.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/auth/STSCredentialsProvider.h>
#include <aws/core/client/RetryStrategy.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>

namespace facebook::velox {

bool inline isS3File(const std::string_view filename) {
  return (filename.substr(0, 2) == "s3" || filename.substr(0, 3) == "s3a");
}

class S3ReadFile final : public ReadFile {
 public:
  explicit S3ReadFile(std::string_view path) : path_(path) {}
  void init(std::shared_ptr<Aws::S3::S3Client> client) {
    client_ = client;
  }
  std::string_view pread(uint64_t offset, uint64_t length, Arena* arena)
      const final;
  std::string_view pread(uint64_t offset, uint64_t length, void* buf)
      const final;
  std::string pread(uint64_t offset, uint64_t length) const final;
  uint64_t size() const final;
  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers) final;
  uint64_t memoryUsage() const final;

 private:
  void preadInternal(uint64_t offset, uint64_t length, char* pos) const;
  std::shared_ptr<Aws::S3::S3Client> client_;
  std::string path_;
  bool closed_ = false;
  int64_t pos_ = 0;
  int64_t content_length_ = 0;
};

// Implementation of S3 FileSystem
class S3FileSystem : public FileSystem {
 public:
  S3FileSystem(std::shared_ptr<const Config> config) : FileSystem(config) {}
  ~S3FileSystem() {}
  void init();
  virtual std::string name() const override {
    return "S3";
  }
  virtual std::unique_ptr<ReadFile> openReadFile(
      std::string_view path) override {
    return std::make_unique<S3ReadFile>(path);
  }
  virtual std::unique_ptr<WriteFile> openWriteFile(
      std::string_view path) override {
    // Not yet implemented
    return nullptr;
  }
  // Configure default AWS credentials provider chain.
  void configureDefaultCredentialChain();
  // Configure with access and secret keys. Used for on-prem.
  void configureAccessKey(
      const std::string& access_key,
      const std::string& secret_key,
      const std::string& session_token = "");
  // get API
  std::string getAccessKey() const;
  std::string getSecretKey() const;
  std::string getSessionToken() const;
  std::string getEndPoint() const {
    return endpoint_;
  }
  std::string getScheme() const {
    return scheme_;
  }
  std::string getRegion() const {
    return region_;
  }

 private:
  std::string scheme_ = "https";
  std::string endpoint_;
  std::string region_;
  Aws::Client::ClientConfiguration client_config_;
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> credentials_provider_;
  std::shared_ptr<Aws::S3::S3Client> client_;
};

} // namespace facebook::velox
