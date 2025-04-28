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
#include "velox/dwio/parquet/crypto/KMSClient.h"
#include <glog/logging.h>
#include <cstddef>
#include <map>
#include "velox/common/base/Exceptions.h"
#include "velox/common/encode/Base64.h"
#include "velox/dwio/parquet/crypto/Exception.h"
#include "velox/dwio/parquet/crypto/KeyMetadataAssembler.h"
#include "velox/dwio/parquet/crypto/Utils.h"

namespace facebook::velox::parquet {

const std::string EncryptedKeyVersionEEK = "EEK";

std::shared_ptr<EncryptedKeyVersion> KMSClient::parseKeyMetadata(
    const std::string& keyMetadata) {
  KeyMetadata unAssembledKeyMetadata =
      KeyMetadataAssembler::unAssembly(keyMetadata);

  std::shared_ptr<KeyVersion> keyVersion = std::make_shared<KeyVersion>(
      "", EncryptedKeyVersionEEK, unAssembledKeyMetadata.eek);

  std::string versionName = unAssembledKeyMetadata.name + "@" +
      std::to_string(unAssembledKeyMetadata.version);
  std::shared_ptr<EncryptedKeyVersion> encryptedKeyVersion =
      std::make_shared<EncryptedKeyVersion>(
          unAssembledKeyMetadata.name,
          versionName,
          unAssembledKeyMetadata.iv,
          keyVersion);

  VELOX_USER_CHECK(
      !encryptedKeyVersion->encryptionKeyVersionName.empty(),
      "[CLAC] encryptionKeyVersionName empty");
  VELOX_USER_CHECK(
      !encryptedKeyVersion->encryptedKeyIv.empty(),
      "[CLAC] encryptedKeyIv empty");
  VELOX_USER_CHECK(
      encryptedKeyVersion->encryptedKeyVersion->versionName ==
          EncryptedKeyVersionEEK,
      "[CLAC] encryptedKey version name must be '" + EncryptedKeyVersionEEK +
          "', is '" + encryptedKeyVersion->encryptedKeyVersion->versionName +
          "'");
  VELOX_USER_CHECK(
      encryptedKeyVersion->encryptedKeyVersion,
      "[CLAC] encryptedKeyVersion is null");
  return encryptedKeyVersion;
}

std::string KMSClient::getKey(
    const std::string& keyMetadata,
    const std::string& doAs) {
  std::shared_ptr<EncryptedKeyVersion> encryptedKeyVersion =
      parseKeyMetadata(keyMetadata);
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
    if (error.find("http status code 403") != std::string::npos ||
        error.find("http status code 404") != std::string::npos ||
        error.find("no keyversion exists for key ") != std::string::npos ||
        error.find(" not found") != std::string::npos) {
      exceptionCache_.set(cacheKey, error, 300);
    }
    throw;
  }

  cache_.set(cacheKey, decryptedKey, 300); // 5 minutes
  return decryptedKey;
}

} // namespace facebook::velox::parquet
