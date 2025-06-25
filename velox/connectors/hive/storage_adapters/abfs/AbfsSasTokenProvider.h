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

#include "velox/common/config/Config.h"

#include <string>

namespace facebook::velox::filesystems {

/// SAS permissions reference:
/// https://learn.microsoft.com/en-us/rest/api/storageservices/create-service-sas#permissions-for-a-directory-container-or-blob
///
/// ReadClient uses "read" permission for Download and GetProperties.
/// WriteClient uses "read" permission for GetProperties, and "write" permission
/// for other operations.
static const std::string kAbfsReadOperation{"read"};
static const std::string kAbfsWriteOperation{"write"};

/// Interface for providing SAS tokens for ABFS file system operations.
/// Adapted from the Hadoop Azure implementation:
/// org.apache.hadoop.fs.azurebfs.extensions.SASTokenProvider
class AbfsSasTokenProvider {
 public:
  virtual ~AbfsSasTokenProvider() = default;

  virtual void initialize(
      const config::ConfigBase& conf,
      const std::string& accountNameWithSuffix) = 0;

  virtual std::string getSasToken(
      const std::string& fileSystem,
      const std::string& path,
      const std::string& operation) = 0;
};
}; // namespace facebook::velox::filesystems
