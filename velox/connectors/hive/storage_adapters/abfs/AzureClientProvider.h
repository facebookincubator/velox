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

#include "velox/connectors/hive/storage_adapters/abfs/AzureBlobClient.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureDataLakeFileClient.h"

namespace facebook::velox::filesystems {

/// Provider interface for creating Azure Blob and Data Lake clients.
class AzureClientProvider {
 public:
  virtual ~AzureClientProvider() = default;

  explicit AzureClientProvider(const std::shared_ptr<AbfsPath>& path)
      : abfsPath_(path) {}

  virtual std::unique_ptr<AzureBlobClient> getBlobClient() = 0;

  virtual std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient() = 0;

 protected:
  std::shared_ptr<AbfsPath> abfsPath_;
};

} // namespace facebook::velox::filesystems
