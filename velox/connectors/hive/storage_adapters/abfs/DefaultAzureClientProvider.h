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

#include <azure/identity/client_secret_credential.hpp>

#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h"

namespace facebook::velox::filesystems {

class DefaultAzureClientProvider final : public AzureClientProvider {
 public:
  DefaultAzureClientProvider(
      const std::shared_ptr<AbfsPath>& abfsPath,
      const config::ConfigBase& config);

  std::unique_ptr<AzureBlobClient> getBlobClient() override;

  std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient() override;

  /// Test only.
  std::string connectionString() const {
    return connectionString_;
  }

  /// Test only.
  std::string tenentId() const {
    return tenentId_;
  }

  /// Test only.
  std::string authorityHost() const {
    return authorityHost_;
  }

  /// Test only.
  static void setUpTestWriteClient(
      std::function<std::unique_ptr<AzureDataLakeFileClient>()> testClientFn) {
    testWriteClientFn_ = testClientFn;
  }

  /// Test only.
  static void tearDownTestWriteClient() {
    testWriteClientFn_ = nullptr;
  }

 private:
  void initAuthType(const config::ConfigBase& config);

  std::string authType_{};

  // Container name is called FileSystem in some Azure API.
  std::string connectionString_;

  std::string sas_;

  std::string tenentId_;
  std::string authorityHost_;
  std::shared_ptr<Azure::Core::Credentials::TokenCredential> tokenCredential_;

  static std::function<std::unique_ptr<AzureDataLakeFileClient>()>
      testWriteClientFn_;
};

} // namespace facebook::velox::filesystems
