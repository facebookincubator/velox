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

#include <azure/core/credentials/credentials.hpp>
#include <folly/hash/Hash.h>
#include <string>

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::filesystems {

// This is used to specify the Azurite endpoint in testing.
static std::string kAzureBlobEndpoint{"fs.azure.blob-endpoint"};

// The authentication mechanism is set in `fs.azure.account.auth.type` (or the
// account specific variant). The supported values are SharedKey, OAuth and SAS.
static std::string kAzureAccountAuthType{"fs.azure.account.auth.type"};

static std::string kAzureAccountKey{"fs.azure.account.key"};

static std::string kAzureSASKey{"fs.azure.sas.fixed.token"};

static std::string kAzureAccountOAuth2ClientId{
    "fs.azure.account.oauth2.client.id"};
static std::string kAzureAccountOAuth2ClientSecret{
    "fs.azure.account.oauth2.client.secret"};

// Token end point, this can be found through Azure portal. For example:
// https://login.microsoftonline.com/{TENANTID}/oauth2/token
static std::string kAzureAccountOAuth2ClientEndpoint{
    "fs.azure.account.oauth2.client.endpoint"};

class AbfsConfig {
 public:
  explicit AbfsConfig(
      std::string_view path,
      const config::ConfigBase& config,
      bool initDfsClient);

  std::string authType() const {
    return authType_;
  }

  std::string fileSystem() const {
    return fileSystem_;
  }

  std::string filePath() const {
    return filePath_;
  }

  std::string connectionString() const {
    return connectionString_;
  }

  std::string url() const {
    return url_;
  }

  std::string urlWithSasToken() const {
    return urlWithSasToken_;
  }

  std::shared_ptr<Azure::Core::Credentials::TokenCredential> tokenCredential()
      const {
    return tokenCredential_;
  }

  std::string tenentId() const {
    return tenentId_;
  }

  std::string authorityHost() const {
    return authorityHost_;
  }

 private:
  // Container name is called FileSystem in some Azure API.
  std::string fileSystem_;
  std::string filePath_;
  std::string authType_;
  std::string connectionString_;
  std::string urlWithSasToken_;
  std::string url_;
  std::string tenentId_;
  std::string authorityHost_;
  std::shared_ptr<Azure::Core::Credentials::TokenCredential> tokenCredential_;
};

} // namespace facebook::velox::filesystems
