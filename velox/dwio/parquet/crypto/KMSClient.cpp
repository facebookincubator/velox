#include "KMSClient.h"
#include <map>
#include "KeyMetadataAssembler.h"
#include "common/base/Exceptions.h"
#include "common/encode/Base64.h"
#include <glog/logging.h>
#include "Exception.h"
#include <cstddef>
#include "Utils.h"

namespace facebook::velox::parquet {

const std::string EncryptedKeyVersionEEK = "EEK";

std::shared_ptr<EncryptedKeyVersion> KMSClient::parseKeyMetadata(const std::string& keyMetadata) {
  KeyMetadata unAssembledKeyMetadata = KeyMetadataAssembler::unAssembly(keyMetadata);

  std::shared_ptr<KeyVersion> keyVersion = std::make_shared<KeyVersion>("", EncryptedKeyVersionEEK, unAssembledKeyMetadata.eek);

  std::string versionName = unAssembledKeyMetadata.name + "@" + std::to_string(unAssembledKeyMetadata.version);
  std::shared_ptr<EncryptedKeyVersion> encryptedKeyVersion = std::make_shared<EncryptedKeyVersion>(
    unAssembledKeyMetadata.name, versionName, unAssembledKeyMetadata.iv, keyVersion);

  VELOX_USER_CHECK(!encryptedKeyVersion->encryptionKeyVersionName.empty(), "[CLAC] encryptionKeyVersionName empty");
  VELOX_USER_CHECK(!encryptedKeyVersion->encryptedKeyIv.empty(), "[CLAC] encryptedKeyIv empty");
  VELOX_USER_CHECK(encryptedKeyVersion->encryptedKeyVersion->versionName == EncryptedKeyVersionEEK,
                   "[CLAC] encryptedKey version name must be '" + EncryptedKeyVersionEEK + "', is '" + encryptedKeyVersion->encryptedKeyVersion->versionName + "'");
  VELOX_USER_CHECK(encryptedKeyVersion->encryptedKeyVersion, "[CLAC] encryptedKeyVersion is null");
  return encryptedKeyVersion;
}

std::string KMSClient::getKey(const std::string& keyMetadata, const std::string& doAs) {
  std::shared_ptr<EncryptedKeyVersion> encryptedKeyVersion = parseKeyMetadata(keyMetadata);
  CacheableEncryptedKeyVersion cacheKey{doAs, encryptedKeyVersion};

  std::optional<std::string> decryptedKeyOpt = cache_.get(cacheKey);
  if (decryptedKeyOpt.has_value()) {
    return decryptedKeyOpt.value();
  }
  std::optional<std::string> exceptionOpt = exceptionCache_.get(cacheKey);
  if (exceptionOpt.has_value()) {
    throw CryptoException(exceptionOpt.value());
  }

  std::string decryptedKey{""};
  try {
    decryptedKey = decryptKey(encryptedKeyVersion, doAs).material;
  } catch (const CryptoException& e) {
    std::string error = e.what();
    if (error.find("http status code 403") != std::string::npos
        || error.find("http status code 404") != std::string::npos
        || error.find("no keyversion exists for key ") != std::string::npos
        || error.find(" not found") != std::string::npos) {
      exceptionCache_.set(cacheKey, error, 300);
    }
    throw;
  }

  cache_.set(cacheKey, decryptedKey, 300); // 5 minutes
  return decryptedKey;
}

}
