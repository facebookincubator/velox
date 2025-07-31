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
#include "velox/connectors/hive/storage_adapters/abfs/AbfsConfig.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProvider.h"

namespace facebook::velox::filesystems {

/// AzureClientProvider for Shared Key authentication.
class SharedKeyAzureClientProvider final : public AzureClientProvider {
 public:
  SharedKeyAzureClientProvider(
      const std::shared_ptr<AbfsPath>& abfsPath,
      const config::ConfigBase& config);

  std::unique_ptr<AzureBlobClient> getBlobClient() override;

  std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient() override;

  /// Test only.
  std::string connectionString() const {
    return connectionString_;
  }

 private:
  // Container name is called FileSystem in some Azure API.
  std::string connectionString_;
};

/// AzureClientProvider for OAuth authentication.
class OAuthAzureClientProvider final : public AzureClientProvider {
 public:
  OAuthAzureClientProvider(
      const std::shared_ptr<AbfsPath>& abfsPath,
      const config::ConfigBase& config);

  std::unique_ptr<AzureBlobClient> getBlobClient() override;

  std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient() override;

  /// Test only.
  std::string tenentId() const {
    return tenentId_;
  }

  /// Test only.
  std::string authorityHost() const {
    return authorityHost_;
  }

 private:
  std::string tenentId_;
  std::string authorityHost_;
  std::shared_ptr<Azure::Core::Credentials::TokenCredential> tokenCredential_;
};

/// AzureClientProvider for SAS authentication with a fixed SAS token.
class FixedSasAzureClientProvider final : public AzureClientProvider {
 public:
  FixedSasAzureClientProvider(
      const std::shared_ptr<AbfsPath>& abfsPath,
      const config::ConfigBase& config);

  std::unique_ptr<AzureBlobClient> getBlobClient() override;

  std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient() override;

 private:
  std::string sas_;
};

} // namespace facebook::velox::filesystems
