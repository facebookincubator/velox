#include "FileDecryptionProperties.h"
#include "common/base/Exceptions.h"
#include "velox/dwio/parquet/crypto/AesEncryption.h"

namespace facebook::velox::parquet {

FileDecryptionProperties::Builder*
FileDecryptionProperties::Builder::keyRetriever(
    const std::shared_ptr<DecryptionKeyRetriever>& keyRetriever) {
  if (keyRetriever == nullptr)
    return this;

  keyRetriever_ = keyRetriever;
  return this;
}

FileDecryptionProperties::Builder*
FileDecryptionProperties::Builder::aadPrefix(const std::string& aadPrefix) {
  if (aadPrefix.empty()) {
    return this;
  }
  aadPrefix_ = aadPrefix;
  return this;
}

FileDecryptionProperties::Builder*
FileDecryptionProperties::Builder::aadPrefixVerifier(
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier) {
  if (aadPrefixVerifier == nullptr)
    return this;

  aadPrefixVerifier_ = std::move(aadPrefixVerifier);
  return this;
}

FileDecryptionProperties::FileDecryptionProperties(
    std::shared_ptr<DecryptionKeyRetriever> keyRetriever,
    bool checkPlaintextFooterIntegrity,
    const std::string& aadPrefix,
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier,
    bool plaintextFilesAllowed) {
  DCHECK(nullptr != keyRetriever);
  aadPrefixVerifier_ = std::move(aadPrefixVerifier);
  checkPlaintextFooterIntegrity_ = checkPlaintextFooterIntegrity;
  keyRetriever_ = std::move(keyRetriever);
  aadPrefix_ = aadPrefix;
  plaintextFilesAllowed_ = plaintextFilesAllowed;
}

}
