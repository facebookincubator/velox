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

#include "velox/connectors/hive/storage_adapters/abfs/DefaultAzureClientProvider.h"

namespace facebook::velox::filesystems {
namespace {

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

class BlobClientWrapper : public AzureBlobClient {
 public:
  BlobClientWrapper(std::unique_ptr<Azure::Storage::Blobs::BlobClient> client) {
    blobClient_ = std::move(client);
  }

  Azure::Response<Azure::Storage::Blobs::Models::BlobProperties> getProperties()
      override {
    return blobClient_->GetProperties();
  }

  Azure::Response<Azure::Storage::Blobs::Models::DownloadBlobResult> download(
      const Azure::Storage::Blobs::DownloadBlobOptions& options) override {
    return blobClient_->Download(options);
  }

  std::string getUrl() override {
    return blobClient_->GetUrl();
  }

 private:
  std::unique_ptr<Azure::Storage::Blobs::BlobClient> blobClient_;
};

} // namespace

std::function<std::unique_ptr<AzureDataLakeFileClient>()>
    DefaultAzureClientProvider::testWriteClientFn_;

DefaultAzureClientProvider::DefaultAzureClientProvider(
    const std::shared_ptr<AbfsPath>& abfsPath,
    const config::ConfigBase& config)
    : AzureClientProvider(abfsPath) {
  initAuthType(config);
}

std::unique_ptr<AzureBlobClient> DefaultAzureClientProvider::getBlobClient() {
  std::unique_ptr<BlobClient> client;
  if (authType_ == kAzureSASAuthType) {
    auto url = abfsPath_->getUrl(true);
    client = std::make_unique<BlobClient>(fmt::format("{}?{}", url, sas_));
  } else if (authType_ == kAzureOAuthAuthType) {
    auto url = abfsPath_->getUrl(true);
    client = std::make_unique<BlobClient>(url, tokenCredential_);
  } else {
    client =
        std::make_unique<BlobClient>(BlobClient::CreateFromConnectionString(
            connectionString_, abfsPath_->fileSystem(), abfsPath_->filePath()));
  }
  return std::make_unique<BlobClientWrapper>(std::move(client));
}

std::unique_ptr<AzureDataLakeFileClient>
DefaultAzureClientProvider::getDataLakeFileClient() {
  if (testWriteClientFn_) {
    return testWriteClientFn_();
  }

  std::unique_ptr<DataLakeFileClient> client;
  if (authType_ == kAzureSASAuthType) {
    auto url = abfsPath_->getUrl(false);
    client =
        std::make_unique<DataLakeFileClient>(fmt::format("{}?{}", url, sas_));
  } else if (authType_ == kAzureOAuthAuthType) {
    auto url = abfsPath_->getUrl(false);
    client = std::make_unique<DataLakeFileClient>(url, tokenCredential_);
  } else {
    client = std::make_unique<DataLakeFileClient>(
        DataLakeFileClient::CreateFromConnectionString(
            connectionString_, abfsPath_->fileSystem(), abfsPath_->filePath()));
  }
  return std::make_unique<DataLakeFileClientWrapper>(std::move(client));
}

void DefaultAzureClientProvider::initAuthType(
    const config::ConfigBase& config) {
  auto authTypeKey = fmt::format(
      "{}.{}", kAzureAccountAuthType, abfsPath_->accountNameWithSuffix());
  authType_ = kAzureSharedKeyAuthType;
  if (config.valueExists(authTypeKey)) {
    authType_ = config.get<std::string>(authTypeKey).value();
  }
  if (authType_ == kAzureSharedKeyAuthType) {
    auto credKey = fmt::format(
        "{}.{}", kAzureAccountKey, abfsPath_->accountNameWithSuffix());
    VELOX_USER_CHECK(
        config.valueExists(credKey), "Config {} not found", credKey);
    auto firstDot = abfsPath_->accountNameWithSuffix().find_first_of(".");
    auto endpointSuffix =
        abfsPath_->accountNameWithSuffix().substr(firstDot + 5 /* .dfs. */);
    std::stringstream ss;
    ss << "DefaultEndpointsProtocol="
       << (abfsPath_->isHttps() ? "https" : "http");
    ss << ";AccountName=" << abfsPath_->accountName();
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
        "{}.{}",
        kAzureAccountOAuth2ClientId,
        abfsPath_->accountNameWithSuffix());
    auto clientSecretKey = fmt::format(
        "{}.{}",
        kAzureAccountOAuth2ClientSecret,
        abfsPath_->accountNameWithSuffix());
    auto clientEndpointKey = fmt::format(
        "{}.{}",
        kAzureAccountOAuth2ClientEndpoint,
        abfsPath_->accountNameWithSuffix());
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
    auto sasKey =
        fmt::format("{}.{}", kAzureSASKey, abfsPath_->accountNameWithSuffix());
    VELOX_USER_CHECK(config.valueExists(sasKey), "Config {} not found", sasKey);
    sas_ = config.get<std::string>(sasKey).value();
  } else {
    VELOX_USER_FAIL(
        "Unsupported auth type {}, supported auth types are SharedKey, OAuth and SAS.",
        authType_);
  }
}

} // namespace facebook::velox::filesystems
