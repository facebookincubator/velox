#pragma once
#include "FileDecryptionProperties.h"
#include "KeyRetriever.h"
#include "common/base/Exceptions.h"

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
