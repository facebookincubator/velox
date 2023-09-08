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
#include <azure/storage/blobs.hpp>
#include "velox/common/file/File.h"

#include <fmt/format.h>
#include <regex>

namespace facebook::velox::filesystems::abfs {
namespace {
constexpr std::string_view kAbfsScheme{"abfs://"};
constexpr std::string_view kAbfssScheme{"abfss://"};
} // namespace

inline bool isAbfsFile(const std::string_view filename) {
  return filename.find(kAbfsScheme) == 0 || filename.find(kAbfssScheme) == 0;
}

class AbfsAccount {
 public:
  AbfsAccount(const std::string path) {
    auto file = std::string("");
    if (path.find(kAbfssScheme) == 0) {
      file = std::string(path.substr(8));
      scheme_ = kAbfssScheme.substr(0, 5);
    } else {
      file = std::string(path.substr(7));
      scheme_ = kAbfsScheme.substr(0, 4);
    }

    auto firstAt = file.find_first_of("@");
    fileSystem_ = std::string(file.substr(0, firstAt));
    auto firstSep = file.find_first_of("/");
    filePath_ = std::string(file.substr(firstSep + 1));

    accountNameWithSuffix_ = file.substr(firstAt + 1, firstSep - firstAt - 1);
    auto firstDot = accountNameWithSuffix_.find_first_of(".");
    accountName_ = accountNameWithSuffix_.substr(0, firstDot);
    endpointSuffix_ = accountNameWithSuffix_.substr(firstDot + 5);
    credKey_ = fmt::format("fs.azure.account.key.{}", accountNameWithSuffix_);
  }

  const std::string accountNameWithSuffix() {
    return accountNameWithSuffix_;
  }

  const std::string scheme() {
    return scheme_;
  }

  const std::string accountName() {
    return accountName_;
  }

  const std::string endpointSuffix() {
    return endpointSuffix_;
  }

  const std::string fileSystem() {
    return fileSystem_;
  }

  const std::string filePath() {
    return filePath_;
  }

  const std::string credKey() {
    return credKey_;
  }

  std::string connectionString(bool useHttps, std::string accountKey) {
    auto protocol = "https";
    if (!useHttps) {
      // This is only used for testing.
      protocol = "http";
    }

    return fmt::format(
        "DefaultEndpointsProtocol={};AccountName={};AccountKey={};EndpointSuffix={}",
        protocol,
        accountName(),
        accountKey,
        endpointSuffix());
  }

 private:
  std::string scheme_;
  std::string accountName_;
  std::string endpointSuffix_;
  std::string accountNameWithSuffix_;
  std::string fileSystem_;
  std::string filePath_;
  std::string path_;
  std::string credKey_;
};

inline std::string throwStorageExceptionWithOperationDetails(
    std::string operation,
    std::string path,
    Azure::Storage::StorageException& error) {
  VELOX_FAIL(
      "Operation '{}' to path '{}' encountered azure storage exception, Details: '{}'.",
      operation,
      path,
      error.what());
}
} // namespace facebook::velox::filesystems::abfs
