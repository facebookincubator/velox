#pragma once
#include <string>
#include <map>
#include <memory>
#include "velox/dwio/parquet/crypto/AesEncryption.h"
#include "velox/dwio/parquet/crypto/ColumnPath.h"
#include "velox/dwio/parquet/crypto/KeyRetriever.h"

namespace facebook::velox::parquet {

static constexpr ParquetCipher::type kDefaultEncryptionAlgorithm =
    ParquetCipher::AES_GCM_V1;
static constexpr int32_t kMaximalAadMetadataLength = 256;
static constexpr bool kDefaultEncryptedFooter = true;
static constexpr bool kDefaultCheckSignature = true;
static constexpr bool kDefaultAllowPlaintextFiles = false;
static constexpr int32_t kAadFileUniqueLength = 8;

class AADPrefixVerifier {
 public:
  virtual void Verify(const std::string& aad_prefix) = 0;
  virtual ~AADPrefixVerifier() {}
};

class FileDecryptionProperties {
 public:
  class Builder {
   public:
    Builder() {
      checkPlaintextFooterIntegrity_ = kDefaultCheckSignature;
      plaintextFilesAllowed_ = kDefaultAllowPlaintextFiles;
    }
    Builder* keyRetriever(const std::shared_ptr<DecryptionKeyRetriever>& keyRetriever);
    Builder* disableFooterSignatureVerification() {
      checkPlaintextFooterIntegrity_ = false;
      return this;
    }

    Builder* aadPrefix(const std::string& aadPrefix);

    Builder* aadPrefixVerifier(std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier);

    Builder* plaintextFilesAllowed() {
      plaintextFilesAllowed_ = true;
      return this;
    }

    std::unique_ptr<FileDecryptionProperties> build() {
      return std::unique_ptr<FileDecryptionProperties>(new FileDecryptionProperties(
          keyRetriever_, checkPlaintextFooterIntegrity_, aadPrefix_,
          aadPrefixVerifier_, plaintextFilesAllowed_));
    }

   private:
    std::string aadPrefix_;
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier_;

    std::shared_ptr<DecryptionKeyRetriever> keyRetriever_;
    bool checkPlaintextFooterIntegrity_;
    bool plaintextFilesAllowed_;
  };

  std::string aadPrefix() const { return aadPrefix_; }

  const std::shared_ptr<DecryptionKeyRetriever>& keyRetriever() const {
    return keyRetriever_;
  }

  bool checkPlaintextFooterIntegrity() const {
    return checkPlaintextFooterIntegrity_;
  }

  bool plaintextFilesAllowed() const { return plaintextFilesAllowed_; }

  const std::shared_ptr<AADPrefixVerifier>& aadPrefixVerifier() const {
    return aadPrefixVerifier_;
  }

 private:
  std::string aadPrefix_;
  std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier_;

  std::shared_ptr<DecryptionKeyRetriever> keyRetriever_;
  bool checkPlaintextFooterIntegrity_;
  bool plaintextFilesAllowed_;

  FileDecryptionProperties(
      std::shared_ptr<DecryptionKeyRetriever> keyRetriever,
      bool checkPlaintextFooterIntegrity,
      const std::string& aadPrefix,
      std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier,
      bool plaintextFilesAllowed);
};
}
