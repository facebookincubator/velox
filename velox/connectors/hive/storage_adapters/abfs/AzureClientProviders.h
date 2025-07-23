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

#include <memory>

#include "AbfsConfig.h"
#include "AzureBlobClient.h"
#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureBlobClient.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureDataLakeFileClient.h"

namespace facebook::velox::filesystems {

class AzureClientProvider {
 public:
  virtual void initialize(
      const config::ConfigBase& conf,
      const std::string& accountNameWithSuffix) = 0;

  virtual std::unique_ptr<AzureBlobClient> getBlobClient(
      const std::string& fileUrl,
      const std::string& fileSystem,
      const std::string& filePath) = 0;

  virtual std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient(
      const std::string& fileUrl,
      const std::string& fileSystem,
      const std::string& filePath) = 0;
};

class AzureClientProviders {
 public:
  static void registerProvider(
      const std::string& account,
      std::shared_ptr<AzureClientProvider> provider);

  static bool providerRegistered(const std::string& account);

  static std::unique_ptr<AzureBlobClient> getBlobClient(
      const AbfsConfig& config);

  static std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient(
      const AbfsConfig& config);
};

} // namespace facebook::velox::filesystems
