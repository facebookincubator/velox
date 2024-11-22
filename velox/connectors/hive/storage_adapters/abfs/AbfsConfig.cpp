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

#include "velox/connectors/hive/storage_adapters/abfs/AbfsConfig.h"

#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"

#include <azure/identity/client_secret_credential.hpp>

namespace facebook::velox::filesystems {

AbfsConfig::AbfsConfig(
    std::string_view path,
    const config::ConfigBase& config,
    bool initDfsClient) {
  std::string_view file;
  bool isHttps = true;
  if (path.find(kAbfssScheme) == 0) {
    file = path.substr(kAbfssScheme.size());
  } else if (path.find(kAbfsScheme) == 0) {
    file = path.substr(kAbfsScheme.size());
    isHttps = false;
  } else {
    VELOX_FAIL("Invalid ABFS Path {}", path);
  }

  auto firstAt = file.find_first_of("@");
  fileSystem_ = file.substr(0, firstAt);
  auto firstSep = file.find_first_of("/");
  filePath_ = file.substr(firstSep + 1);
  auto accountNameWithSuffix = file.substr(firstAt + 1, firstSep - firstAt - 1);
  std::string accountNameWithSuffixForUrl(accountNameWithSuffix);
  if (!initDfsClient) {
    // We should use correct suffix for blob client.
    size_t start_pos = accountNameWithSuffixForUrl.find("dfs");
    if (start_pos != std::string::npos) {
      accountNameWithSuffixForUrl.replace(start_pos, 3, "blob");
    }
  }

  url_ = fmt::format(
      "{}{}/{}/{}",
      isHttps ? "https://" : "http://",
      accountNameWithSuffixForUrl,
      fileSystem_,
      filePath_);

  auto authTypeKey =
      fmt::format("{}.{}", kAzureAccountAuthType, accountNameWithSuffix);
  authType_ = "SharedKey";
  if (config.valueExists(authTypeKey)) {
    authType_ = config.get<std::string>(authTypeKey).value();
  }
  if (authType_ == "SharedKey") {
    auto credKey =
        fmt::format("{}.{}", kAzureAccountKey, accountNameWithSuffix);
    VELOX_USER_CHECK(
        config.valueExists(credKey), "Config {} not found", credKey);
    auto firstDot = accountNameWithSuffix.find_first_of(".");
    auto accountName = accountNameWithSuffix.substr(0, firstDot);
    auto endpointSuffix = accountNameWithSuffix.substr(firstDot + 5);
    std::stringstream ss;
    ss << "DefaultEndpointsProtocol=" << (isHttps ? "https" : "http");
    ss << ";AccountName=" << accountName;
    ss << ";AccountKey=" << config.get<std::string>(credKey).value();
    ss << ";EndpointSuffix=" << endpointSuffix;

    if (config.valueExists(kAzureBlobEndpoint)) {
      ss << ";BlobEndpoint="
         << config.get<std::string>(kAzureBlobEndpoint).value();
    }
    ss << ";";
    connectionString_ = ss.str();
  } else if (authType_ == "OAuth") {
    auto clientIdKey = fmt::format(
        "{}.{}", kAzureAccountOAuth2ClientId, accountNameWithSuffix);
    auto clientSecretKey = fmt::format(
        "{}.{}", kAzureAccountOAuth2ClientSecret, accountNameWithSuffix);
    auto clientEndpointKey = fmt::format(
        "{}.{}", kAzureAccountOAuth2ClientEndpoint, accountNameWithSuffix);
    VELOX_USER_CHECK(
        config.valueExists(clientIdKey), "Config {} not found", clientIdKey);
    VELOX_USER_CHECK(
        config.valueExists(clientSecretKey),
        "Config {} not found",
        clientSecretKey);
    VELOX_USER_CHECK(
        config.valueExists(clientEndpointKey),
        "Config {} not found",
        clientEndpointKey);
    auto clientEndpoint = config.get<std::string>(clientEndpointKey).value();
    auto firstSep = clientEndpoint.find_first_of("/", /* https:// */ 8);
    authorityHost_ = clientEndpoint.substr(0, firstSep + 1);
    auto sedondSep = clientEndpoint.find_first_of("/", firstSep + 1);
    tenentId_ = clientEndpoint.substr(firstSep + 1, sedondSep - firstSep - 1);
    Azure::Identity::ClientSecretCredentialOptions options;
    options.AuthorityHost = authorityHost_;
    tokenCredential_ =
        std::make_shared<Azure::Identity::ClientSecretCredential>(
            tenentId_,
            config.get<std::string>(clientIdKey).value(),
            config.get<std::string>(clientSecretKey).value(),
            options);
  } else if (authType_ == "SAS") {
    auto sasKey = fmt::format("{}.{}", kAzureSASKey, accountNameWithSuffix);
    VELOX_USER_CHECK(config.valueExists(sasKey), "Config {} not found", sasKey);
    urlWithSasToken_ =
        fmt::format("{}?{}", url_, config.get<std::string>(sasKey).value());
  } else {
    VELOX_USER_FAIL(
        "Unsupported auth type {}, supported auth types are SharedKey, OAuth and SAS.",
        authType_);
  }
}

} // namespace facebook::velox::filesystems
