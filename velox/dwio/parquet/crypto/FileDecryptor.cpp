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
#include "velox/dwio/parquet/crypto/FileDecryptor.h"
#include <utility>
#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/crypto/AesEncryption.h"
#include "velox/dwio/parquet/crypto/Exception.h"
#include "velox/dwio/parquet/crypto/FileDecryptionProperties.h"

namespace facebook::velox::parquet {

// Decryptor
Decryptor::Decryptor(
    std::shared_ptr<AesDecryptor> aesDecryptor,
    std::string key,
    std::string fileAad)
    : aesDecryptor_(aesDecryptor),
      key_(std::move(key)),
      fileAad_(std::move(fileAad)) {}

int Decryptor::plaintextLength(int ciphertextLen) const {
  return aesDecryptor_->plaintextLength(ciphertextLen);
}

int Decryptor::ciphertextLength(int plaintextLen) const {
  return aesDecryptor_->ciphertextLength(plaintextLen);
}

int Decryptor::decrypt(
    const uint8_t* ciphertext,
    int ciphertextLen,
    uint8_t* plaintext,
    int plaintextLen,
    std::string_view aad) {
  return aesDecryptor_->decrypt(
      ciphertext,
      ciphertextLen,
      reinterpret_cast<const uint8_t*>(key_.data()),
      static_cast<int>(key_.size()),
      reinterpret_cast<const uint8_t*>(aad.data()),
      static_cast<int>(aad.size()),
      plaintext,
      plaintextLen);
}

FileDecryptor::FileDecryptor(
    FileDecryptionProperties* properties,
    std::string fileAad,
    ParquetCipher::type algorithm,
    std::string user)
    : properties_(properties),
      fileAad_(std::move(fileAad)),
      algorithm_(algorithm),
      user_(std::move(user)) {}

std::shared_ptr<ColumnDecryptionSetup> FileDecryptor::setColumnCryptoMetadata(
    ColumnPath& columnPath,
    bool encrypted,
    std::string& keyMetadata,
    int columnOrdinal) {
  std::shared_ptr<ColumnDecryptionSetup> columnDecryptionSetup;
  std::string columnKey;
  std::string savedException{""};
  if (!encrypted) {
    columnDecryptionSetup = std::make_shared<ColumnDecryptionSetup>(
        columnPath,
        false,
        false,
        nullptr,
        nullptr,
        columnOrdinal,
        savedException);
  } else {
    try {
      columnKey = properties_->keyRetriever()->getKey(keyMetadata, user_);
    } catch (CryptoException& e) {
      std::string error = e.what();
      if (error.find("http status code 403") !=
          std::string::npos) { // KeyAccessDeniedException
        columnKey = "";
        savedException = error;
      } else {
        throw;
      }
    }

    if (columnKey.empty()) {
      columnDecryptionSetup = std::make_shared<ColumnDecryptionSetup>(
          columnPath,
          true,
          false,
          nullptr,
          nullptr,
          columnOrdinal,
          savedException);
    } else {
      columnDecryptionSetup = std::make_shared<ColumnDecryptionSetup>(
          columnPath,
          true,
          true,
          getColumnDataDecryptor(columnKey),
          getColumnMetaDecryptor(columnKey),
          columnOrdinal,
          savedException);
    }
  }
  columnPathToDecryptionSetupMap_[columnPath.toDotString()] =
      columnDecryptionSetup;
  return columnDecryptionSetup;
}

std::shared_ptr<Decryptor> FileDecryptor::getColumnMetaDecryptor(
    const std::string& columnKey) {
  return getColumnDecryptor(columnKey, true);
}

std::shared_ptr<Decryptor> FileDecryptor::getColumnDataDecryptor(
    const std::string& columnKey) {
  return getColumnDecryptor(columnKey, false);
}

std::shared_ptr<Decryptor> FileDecryptor::getColumnDecryptor(
    const std::string& columnKey,
    bool metadata) {
  int key_len = static_cast<int>(columnKey.size());
  auto aesDecryptor = AesDecryptor::make(algorithm_, key_len, metadata);
  return std::make_shared<Decryptor>(
      std::move(aesDecryptor), columnKey, fileAad_);
}

std::string FileDecryptor::handleAadPrefix(
    FileDecryptionProperties* fileDecryptionProperties,
    thrift::EncryptionAlgorithm& encryptionAlgorithm) {
  std::string aadPrefixInProperties = fileDecryptionProperties->aadPrefix();
  std::string aadPrefix = aadPrefixInProperties;

  std::string aadPrefixInAlgo, aadFileUnique;
  bool supplyAadPrefix;
  if (encryptionAlgorithm.__isset.AES_GCM_V1) {
    aadPrefixInAlgo = encryptionAlgorithm.AES_GCM_V1.aad_prefix;
    aadFileUnique = encryptionAlgorithm.AES_GCM_V1.aad_file_unique;
    supplyAadPrefix = encryptionAlgorithm.AES_GCM_V1.supply_aad_prefix;
  } else if (encryptionAlgorithm.__isset.AES_GCM_CTR_V1) {
    aadPrefixInAlgo = encryptionAlgorithm.AES_GCM_CTR_V1.aad_prefix;
    aadFileUnique = encryptionAlgorithm.AES_GCM_CTR_V1.aad_file_unique;
    supplyAadPrefix = encryptionAlgorithm.AES_GCM_CTR_V1.supply_aad_prefix;
  } else {
    VELOX_USER_FAIL("[CLAC] Unsupported algorithm");
  }
  bool fileHasAadPrefix = !aadPrefixInAlgo.empty();
  std::string aadPrefixInFile = aadPrefixInAlgo;

  if (supplyAadPrefix && aadPrefixInProperties.empty()) {
    throw CryptoException(
        "AAD prefix used for file encryption, "
        "but not stored in file and not supplied "
        "in decryption properties");
  }

  if (fileHasAadPrefix) {
    if (!aadPrefixInProperties.empty()) {
      if (aadPrefixInProperties.compare(aadPrefixInFile) != 0) {
        throw CryptoException(
            "AAD Prefix in file and in properties "
            "is not the same");
      }
    }
    aadPrefix = aadPrefixInFile;
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier =
        fileDecryptionProperties->aadPrefixVerifier();
    if (aadPrefixVerifier != nullptr)
      aadPrefixVerifier->Verify(aadPrefix);
  } else {
    if (!supplyAadPrefix && !aadPrefixInProperties.empty()) {
      throw CryptoException(
          "AAD Prefix set in decryption properties, but was not used "
          "for file encryption");
    }
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier =
        fileDecryptionProperties->aadPrefixVerifier();
    if (aadPrefixVerifier != nullptr) {
      throw CryptoException(
          "AAD Prefix Verifier is set, but AAD Prefix not found in file");
    }
  }
  return aadPrefix + aadFileUnique;
}

ParquetCipher::type FileDecryptor::getEncryptionAlgorithm(
    thrift::EncryptionAlgorithm& encryptionAlgorithm) {
  ParquetCipher::type algo;
  if (encryptionAlgorithm.__isset.AES_GCM_V1) {
    algo = ParquetCipher::type::AES_GCM_V1;
  } else if (encryptionAlgorithm.__isset.AES_GCM_CTR_V1) {
    algo = ParquetCipher::type::AES_GCM_CTR_V1;
  } else {
    VELOX_USER_FAIL("[CLAC] unknown encryption algorithm");
  }
  return algo;
}

} // namespace facebook::velox::parquet
