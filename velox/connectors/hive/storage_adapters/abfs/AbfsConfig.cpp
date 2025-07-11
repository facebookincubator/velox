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
#include <folly/Synchronized.h>
#include <string>
#include <unordered_map>

namespace facebook::velox::filesystems {
namespace {

constexpr int64_t kDefaultSasTokenRenewPeriod = 120; // in seconds

folly::Synchronized<
    std::unordered_map<std::string, AbfsSasTokenProviderFactory>>&
sasTokenProviderFactories() {
  static folly::Synchronized<
      std::unordered_map<std::string, AbfsSasTokenProviderFactory>>
      providers;
  return providers;
}

Azure::DateTime getExpiry(const std::string& token) {
  if (token.empty()) {
    return Azure::DateTime::clock::time_point::min();
  }

  const std::string signedExpiry = "se=";
  const int32_t signedExpiryLen = 3;

  auto start = token.find(signedExpiry);
  if (start == std::string::npos) {
    return Azure::DateTime::clock::time_point::min();
  }
  start += signedExpiryLen;

  auto end = token.find("&", start);
  std::string seValue = (end == std::string::npos)
      ? token.substr(start)
      : token.substr(start, end - start);

  seValue = Azure::Core::Url::Decode(seValue);
  auto seDate =
      Azure::DateTime::Parse(seValue, Azure::DateTime::DateFormat::Rfc3339);

  const std::string signedKeyExpiry = "ske=";
  const int32_t signedKeyExpiryLen = 4;

  start = token.find(signedKeyExpiry);
  if (start == std::string::npos) {
    return seDate;
  }
  start += signedKeyExpiryLen;

  end = token.find("&", start);
  std::string skeValue = (end == std::string::npos)
      ? token.substr(start)
      : token.substr(start, end - start);

  skeValue = Azure::Core::Url::Decode(skeValue);
  auto skeDate =
      Azure::DateTime::Parse(skeValue, Azure::DateTime::DateFormat::Rfc3339);

  return std::min(skeDate, seDate);
}

bool isNearExpiry(Azure::DateTime expiration, int64_t minExpirationInSeconds) {
  if (expiration == Azure::DateTime::clock::time_point::min()) {
    return true;
  }
  auto remaining = std::chrono::duration_cast<std::chrono::seconds>(
                       expiration - Azure::DateTime::clock::now())
                       .count();
  return remaining <= minExpirationInSeconds;
}

} // namespace

std::function<std::unique_ptr<AzureDataLakeFileClient>()>
    AbfsConfig::testWriteClientFn_;

class DataLakeFileClientWrapper final : public AzureDataLakeFileClient {
 public:
  DataLakeFileClientWrapper(std::unique_ptr<DataLakeFileClient> client)
      : client_(std::move(client)) {}

  void create() override {
    client_->Create();
  }

  Azure::Storage::Files::DataLake::Models::PathProperties getProperties()
      override {
    return client_->GetProperties().Value;
  }

  void append(const uint8_t* buffer, size_t size, uint64_t offset) override {
    auto bodyStream = Azure::Core::IO::MemoryBodyStream(buffer, size);
    client_->Append(bodyStream, offset);
  }

  void flush(uint64_t position) override {
    client_->Flush(position);
  }

  void close() override {
    // do nothing.
  }

  std::string getUrl() override {
    return client_->GetUrl();
  }

 private:
  const std::unique_ptr<DataLakeFileClient> client_;
};

class DynamicSasKeyDataLakeFileClient final : public AzureDataLakeFileClient {
 public:
  DynamicSasKeyDataLakeFileClient(
      const std::string& fileUrl,
      const std::string& fileSystem,
      const std::string& filePath,
      const std::shared_ptr<AbfsSasTokenProvider>& sasKeyGenerator,
      int64_t sasTokenRenewPeriod)
      : fileUrl_(fileUrl),
        fileSystem_(fileSystem),
        filePath_(filePath),
        sasKeyGenerator_(sasKeyGenerator),
        sasTokenRenewPeriod_(sasTokenRenewPeriod) {}

  void create() override {
    getWriteClient()->Create();
  }

  Azure::Storage::Files::DataLake::Models::PathProperties getProperties()
      override {
    return getReadClient()->GetProperties().Value;
  }

  void append(const uint8_t* buffer, size_t size, uint64_t offset) override {
    auto bodyStream = Azure::Core::IO::MemoryBodyStream(buffer, size);
    getWriteClient()->Append(bodyStream, offset);
  }

  void flush(uint64_t position) override {
    getWriteClient()->Flush(position);
  }

  void close() override {}

  std::string getUrl() override {
    return getWriteClient()->GetUrl();
  }

 private:
  std::string fileUrl_;
  std::string fileSystem_;
  std::string filePath_;
  std::shared_ptr<AbfsSasTokenProvider> sasKeyGenerator_;
  int64_t sasTokenRenewPeriod_;

  std::unique_ptr<DataLakeFileClient> writeClient_{nullptr};
  Azure::DateTime writeSasExpiration_{
      Azure::DateTime::clock::time_point::min()};

  std::unique_ptr<DataLakeFileClient> readClient_{nullptr};
  Azure::DateTime readSasExpiration_{Azure::DateTime::clock::time_point::min()};

  DataLakeFileClient* getWriteClient() {
    if (writeClient_ == nullptr ||
        isNearExpiry(writeSasExpiration_, sasTokenRenewPeriod_)) {
      const auto sas = sasKeyGenerator_->getSasToken(
          fileSystem_, filePath_, kAbfsWriteOperation);
      writeSasExpiration_ = getExpiry(sas);
      writeClient_ = std::make_unique<DataLakeFileClient>(
          fmt::format("{}?{}", fileUrl_, sas));
    }
    return writeClient_.get();
  }

  DataLakeFileClient* getReadClient() {
    if (readClient_ == nullptr ||
        isNearExpiry(readSasExpiration_, sasTokenRenewPeriod_)) {
      const auto sas = sasKeyGenerator_->getSasToken(
          fileSystem_, filePath_, kAbfsReadOperation);
      readSasExpiration_ = getExpiry(sas);
      readClient_ = std::make_unique<DataLakeFileClient>(
          fmt::format("{}?{}", fileUrl_, sas));
    }
    return readClient_.get();
  }
};

class AzureBlobFileClientWrapper : public AzureBlobClient {
 public:
  AzureBlobFileClientWrapper(
      std::unique_ptr<Azure::Storage::Blobs::BlobClient> client) {
    blobClient_ = std::move(client);
  }

  Azure::Response<Azure::Storage::Blobs::Models::BlobProperties> GetProperties()
      override {
    return blobClient_->GetProperties();
  }

  Azure::Response<Azure::Storage::Blobs::Models::DownloadBlobResult> Download(
      const Azure::Storage::Blobs::DownloadBlobOptions& options) override {
    return blobClient_->Download(options);
  }

  std::string GetUrl() override {
    return blobClient_->GetUrl();
  }

 private:
  std::unique_ptr<Azure::Storage::Blobs::BlobClient> blobClient_;
};

class DynamicSasKeyBlobClient : public AzureBlobClient {
 public:
  DynamicSasKeyBlobClient(
      const std::string& blobUrl,
      const std::string& fileSystem,
      const std::string& filePath,
      const std::shared_ptr<AbfsSasTokenProvider>& sasTokenProvider,
      int64_t sasTokenRenewPeriod)
      : blobUrl_(blobUrl),
        fileSystem_(fileSystem),
        filePath_(filePath),
        sasTokenProvider_(sasTokenProvider),
        sasTokenRenewPeriod_(sasTokenRenewPeriod) {}

  Azure::Response<Azure::Storage::Blobs::Models::BlobProperties> GetProperties()
      override {
    return getBlobClient()->GetProperties();
  }

  Azure::Response<Azure::Storage::Blobs::Models::DownloadBlobResult> Download(
      const Azure::Storage::Blobs::DownloadBlobOptions& options) override {
    return getBlobClient()->Download(options);
  }

  std::string GetUrl() override {
    return getBlobClient()->GetUrl();
  }

 private:
  std::string blobUrl_;
  std::string fileSystem_;
  std::string filePath_;
  std::shared_ptr<AbfsSasTokenProvider> sasTokenProvider_;
  int64_t sasTokenRenewPeriod_;

  std::unique_ptr<Azure::Storage::Blobs::BlobClient> blobClient_{nullptr};
  Azure::DateTime sasExpiration_{Azure::DateTime::clock::time_point::min()};

  BlobClient* getBlobClient() {
    if (blobClient_ == nullptr ||
        isNearExpiry(sasExpiration_, sasTokenRenewPeriod_)) {
      const auto sas = sasTokenProvider_->getSasToken(
          fileSystem_, filePath_, kAbfsReadOperation);
      sasExpiration_ = getExpiry(sas);
      blobClient_ = std::make_unique<Azure::Storage::Blobs::BlobClient>(
          fmt::format("{}?{}", blobUrl_, sas));
    }
    return blobClient_.get();
  }
};

std::unique_ptr<AbfsSasTokenProvider> getSasTokenProvider(
    const std::string& accountName) {
  return sasTokenProviderFactories().withRLock(
      [&](const auto& generators) -> std::unique_ptr<AbfsSasTokenProvider> {
        if (const auto it = generators.find(accountName);
            it != generators.end()) {
          return it->second();
        }
        return nullptr;
      });
}

void registerSasTokenProvider(
    const std::string& accountName,
    const AbfsSasTokenProviderFactory& factory) {
  sasTokenProviderFactories().withWLock([&](auto& generators) {
    if (generators.find(accountName) != generators.end()) {
      VELOX_USER_FAIL(
          "SAS key generator for {} already registered", accountName);
    }
    generators.emplace(accountName, factory);
  });
}

AbfsConfig::AbfsConfig(
    std::string_view path,
    const config::ConfigBase& config) {
  std::string_view file;
  isHttps_ = true;
  if (path.find(kAbfssScheme) == 0) {
    file = path.substr(kAbfssScheme.size());
  } else if (path.find(kAbfsScheme) == 0) {
    file = path.substr(kAbfsScheme.size());
    isHttps_ = false;
  } else {
    VELOX_FAIL("Invalid ABFS Path {}", path);
  }

  auto firstAt = file.find_first_of("@");
  fileSystem_ = file.substr(0, firstAt);
  auto firstSep = file.find_first_of("/");
  filePath_ = file.substr(firstSep + 1);
  accountNameWithSuffix_ = file.substr(firstAt + 1, firstSep - firstAt - 1);
  auto firstDot = accountNameWithSuffix_.find_first_of(".");
  accountName_ = accountNameWithSuffix_.substr(0, firstDot);

  auto authTypeKey =
      fmt::format("{}.{}", kAzureAccountAuthType, accountNameWithSuffix_);
  authType_ = kAzureSharedKeyAuthType;
  if (config.valueExists(authTypeKey)) {
    authType_ = config.get<std::string>(authTypeKey).value();
  }
  if (authType_ == kAzureSharedKeyAuthType) {
    auto credKey =
        fmt::format("{}.{}", kAzureAccountKey, accountNameWithSuffix_);
    VELOX_USER_CHECK(
        config.valueExists(credKey), "Config {} not found", credKey);
    auto endpointSuffix = accountNameWithSuffix_.substr(firstDot + 5);
    std::stringstream ss;
    ss << "DefaultEndpointsProtocol=" << (isHttps_ ? "https" : "http");
    ss << ";AccountName=" << accountName_;
    ss << ";AccountKey=" << config.get<std::string>(credKey).value();
    ss << ";EndpointSuffix=" << endpointSuffix;

    if (config.valueExists(kAzureBlobEndpoint)) {
      ss << ";BlobEndpoint="
         << config.get<std::string>(kAzureBlobEndpoint).value();
    }
    ss << ";";
    connectionString_ = ss.str();
  } else if (authType_ == kAzureOAuthAuthType) {
    auto clientIdKey = fmt::format(
        "{}.{}", kAzureAccountOAuth2ClientId, accountNameWithSuffix_);
    auto clientSecretKey = fmt::format(
        "{}.{}", kAzureAccountOAuth2ClientSecret, accountNameWithSuffix_);
    auto clientEndpointKey = fmt::format(
        "{}.{}", kAzureAccountOAuth2ClientEndpoint, accountNameWithSuffix_);
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
  } else if (authType_ == kAzureSASAuthType) {
    if (sasTokenProvider_ = getSasTokenProvider(accountName_);
        sasTokenProvider_ != nullptr) {
      sasTokenProvider_->initialize(config, accountNameWithSuffix_);
    } else {
      // Use the fixed SAS key if no SAS key generator is registered.
      auto sasKey = fmt::format("{}.{}", kAzureSASKey, accountNameWithSuffix_);
      VELOX_USER_CHECK(
          config.valueExists(sasKey), "Config {} not found", sasKey);
      sas_ = config.get<std::string>(sasKey).value();
    }
    sasTokenRenewPeriod_ = config.get<int64_t>(
        kAzureSasTokenRenewPeriod, kDefaultSasTokenRenewPeriod);
  } else {
    VELOX_USER_FAIL(
        "Unsupported auth type {}, supported auth types are SharedKey, OAuth and SAS.",
        authType_);
  }
}

std::unique_ptr<AzureBlobClient> AbfsConfig::getReadFileClient() {
  if (sasTokenProvider_ != nullptr) {
    return std::make_unique<DynamicSasKeyBlobClient>(
        getUrl(true),
        fileSystem_,
        filePath_,
        sasTokenProvider_,
        sasTokenRenewPeriod_);
  }
  std::unique_ptr<BlobClient> client;
  if (authType_ == kAzureSASAuthType) {
    auto url = getUrl(true);
    client = std::make_unique<BlobClient>(fmt::format("{}?{}", url, sas_));
  } else if (authType_ == kAzureOAuthAuthType) {
    auto url = getUrl(true);
    client = std::make_unique<BlobClient>(url, tokenCredential_);
  } else {
    client =
        std::make_unique<BlobClient>(BlobClient::CreateFromConnectionString(
            connectionString_, fileSystem_, filePath_));
  }
  return std::make_unique<AzureBlobFileClientWrapper>(std::move(client));
}

std::unique_ptr<AzureDataLakeFileClient> AbfsConfig::getWriteFileClient() {
  if (testWriteClientFn_) {
    return testWriteClientFn_();
  }
  if (sasTokenProvider_ != nullptr) {
    return std::make_unique<DynamicSasKeyDataLakeFileClient>(
        getUrl(false),
        fileSystem_,
        filePath_,
        sasTokenProvider_,
        sasTokenRenewPeriod_);
  }
  std::unique_ptr<DataLakeFileClient> client;
  if (authType_ == kAzureSASAuthType) {
    auto url = getUrl(false);
    client =
        std::make_unique<DataLakeFileClient>(fmt::format("{}?{}", url, sas_));
  } else if (authType_ == kAzureOAuthAuthType) {
    auto url = getUrl(false);
    client = std::make_unique<DataLakeFileClient>(url, tokenCredential_);
  } else {
    client = std::make_unique<DataLakeFileClient>(
        DataLakeFileClient::CreateFromConnectionString(
            connectionString_, fileSystem_, filePath_));
  }
  return std::make_unique<DataLakeFileClientWrapper>(std::move(client));
}

std::string AbfsConfig::getUrl(bool withblobSuffix) {
  std::string accountNameWithSuffixForUrl(accountNameWithSuffix_);
  if (withblobSuffix) {
    // We should use correct suffix for blob client.
    size_t start_pos = accountNameWithSuffixForUrl.find("dfs");
    if (start_pos != std::string::npos) {
      accountNameWithSuffixForUrl.replace(start_pos, 3, "blob");
    }
  }
  return fmt::format(
      "{}{}/{}/{}",
      isHttps_ ? "https://" : "http://",
      accountNameWithSuffixForUrl,
      fileSystem_,
      filePath_);
}

} // namespace facebook::velox::filesystems
