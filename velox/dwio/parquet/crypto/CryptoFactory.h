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
#include "velox/dwio/parquet/crypto/FileDecryptionProperties.h"
#include "velox/dwio/parquet/crypto/KeyRetriever.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::parquet {

class CryptoFactory {
 public:
  static void initialize(std::shared_ptr<DecryptionKeyRetriever> kmsClient,
                         const bool clacEnabled) {
    instance_ = std::unique_ptr<CryptoFactory>(
        new CryptoFactory(kmsClient, clacEnabled));
  }
  static CryptoFactory& getInstance() {
    if (!instance_) {
      initialize(nullptr, false);
    }
    return *instance_;
  }

  DecryptionKeyRetriever& getDecryptionKeyRetriever() {
    VELOX_USER_CHECK(kmsClient_, "DecryptionKeyRetriever not provided");
    return *kmsClient_;
  }

  std::shared_ptr<FileDecryptionProperties> getFileDecryptionProperties() {
    return FileDecryptionProperties::Builder().plaintextFilesAllowed()
        ->disableFooterSignatureVerification()
        ->keyRetriever(kmsClient_)
        ->build();
  }

  bool clacEnabled() { return clacEnabled_; }

  ~CryptoFactory() {}

 private:
  CryptoFactory(std::shared_ptr<DecryptionKeyRetriever> kmsClient,
                const bool clacEnabled) : kmsClient_(kmsClient), clacEnabled_(clacEnabled) {}

  static std::unique_ptr<CryptoFactory> instance_;
  std::shared_ptr<DecryptionKeyRetriever> kmsClient_;
  bool clacEnabled_;
};

}
