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
#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureBlobClient.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureDataLakeFileClient.h"

namespace facebook::velox::filesystems {

class AbfsAsyncAuthService;

/// Supplies only the runtime services needed to construct an async read client.
struct AzureAsyncReadContext {
  /// References the caller-owned Blob pipeline options used during creation.
  const Azure::Storage::Blobs::BlobClientOptions& clientOptions;
  /// Shares the runtime-owned bounded authentication service.
  std::shared_ptr<AbfsAsyncAuthService> authService;
};

// Provider interface for creating Azure Blob and Data Lake clients.
class AzureClientProvider {
 public:
  virtual ~AzureClientProvider() = default;

  // Creates AzureBlobClient for file read operations.
  virtual std::unique_ptr<AzureBlobClient> getReadFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) = 0;

  /// Creates an optional AzureBlobClient using caller-supplied options.
  virtual std::unique_ptr<AzureBlobClient> getReadFileClientWithOptions(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config,
      const Azure::Storage::Blobs::BlobClientOptions& options) {
    return nullptr;
  }

  /// Creates an optional async read client using narrow runtime services.
  virtual std::unique_ptr<AzureBlobClient> getReadFileClientForAsync(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config,
      const AzureAsyncReadContext& context) {
    return getReadFileClientWithOptions(path, config, context.clientOptions);
  }

  // Creates AzureDataLakeFileClient for file write operations.
  virtual std::unique_ptr<AzureDataLakeFileClient> getWriteFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) = 0;
};

} // namespace facebook::velox::filesystems
