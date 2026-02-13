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

// Adapted from Apache Arrow.

#include "velox/dwio/parquet/writer/arrow/FileEncryptorInternal.h"
#include "velox/dwio/parquet/writer/arrow/Encryption.h"
#include "velox/dwio/parquet/writer/arrow/EncryptionInternal.h"

namespace facebook::velox::parquet::arrow {

// Encryptor.
Encryptor::Encryptor(
    encryption::AesEncryptor* aesEncryptor,
    const std::string& key,
    const std::string& fileAad,
    const std::string& aad,
    ::arrow::MemoryPool* pool)
    : aesEncryptor_(aesEncryptor),
      key_(key),
      fileAad_(fileAad),
      aad_(aad),
      pool_(pool) {}

int Encryptor::ciphertextSizeDelta() {
  return aesEncryptor_->ciphertextSizeDelta();
}

int Encryptor::encrypt(
    const uint8_t* plaintext,
    int plaintextLen,
    uint8_t* ciphertext) {
  return aesEncryptor_->encrypt(
      plaintext,
      plaintextLen,
      str2bytes(key_),
      static_cast<int>(key_.size()),
      str2bytes(aad_),
      static_cast<int>(aad_.size()),
      ciphertext);
}

// InternalFileEncryptor.
InternalFileEncryptor::InternalFileEncryptor(
    FileEncryptionProperties* properties,
    ::arrow::MemoryPool* pool)
    : properties_(properties), pool_(pool) {
  if (properties_->isUtilized()) {
    throw ParquetException("Re-using encryption properties for another file");
  }
  properties_->setUtilized();
}

void InternalFileEncryptor::wipeOutEncryptionKeys() {
  properties_->wipeOutEncryptionKeys();

  for (auto const& i : allEncryptors_) {
    i->wipeOut();
  }
}

std::shared_ptr<Encryptor> InternalFileEncryptor::getFooterEncryptor() {
  if (footerEncryptor_ != nullptr) {
    return footerEncryptor_;
  }

  ParquetCipher::type algorithm = properties_->algorithm().algorithm;
  std::string footerAad = encryption::createFooterAad(properties_->fileAad());
  std::string footerKey = properties_->footerKey();
  auto aesEncryptor = getMetaAesEncryptor(algorithm, footerKey.size());
  footerEncryptor_ = std::make_shared<Encryptor>(
      aesEncryptor, footerKey, properties_->fileAad(), footerAad, pool_);
  return footerEncryptor_;
}

std::shared_ptr<Encryptor> InternalFileEncryptor::getFooterSigningEncryptor() {
  if (footerSigningEncryptor_ != nullptr) {
    return footerSigningEncryptor_;
  }

  ParquetCipher::type algorithm = properties_->algorithm().algorithm;
  std::string footerAad = encryption::createFooterAad(properties_->fileAad());
  std::string footerSigningKey = properties_->footerKey();
  auto aesEncryptor = getMetaAesEncryptor(algorithm, footerSigningKey.size());
  footerSigningEncryptor_ = std::make_shared<Encryptor>(
      aesEncryptor, footerSigningKey, properties_->fileAad(), footerAad, pool_);
  return footerSigningEncryptor_;
}

std::shared_ptr<Encryptor> InternalFileEncryptor::getColumnMetaEncryptor(
    const std::string& columnPath) {
  return getColumnEncryptor(columnPath, true);
}

std::shared_ptr<Encryptor> InternalFileEncryptor::getColumnDataEncryptor(
    const std::string& columnPath) {
  return getColumnEncryptor(columnPath, false);
}

std::shared_ptr<Encryptor>
InternalFileEncryptor::InternalFileEncryptor::getColumnEncryptor(
    const std::string& columnPath,
    bool metadata) {
  // First look if we already got the encryptor from before.
  if (metadata) {
    if (columnMetadataMap_.find(columnPath) != columnMetadataMap_.end()) {
      return columnMetadataMap_.at(columnPath);
    }
  } else {
    if (columnDataMap_.find(columnPath) != columnDataMap_.end()) {
      return columnDataMap_.at(columnPath);
    }
  }
  auto columnProp = properties_->columnEncryptionProperties(columnPath);
  if (columnProp == nullptr) {
    return nullptr;
  }

  std::string key;
  if (columnProp->isEncryptedWithFooterKey()) {
    key = properties_->footerKey();
  } else {
    key = columnProp->key();
  }

  ParquetCipher::type algorithm = properties_->algorithm().algorithm;
  auto aesEncryptor = metadata ? getMetaAesEncryptor(algorithm, key.size())
                               : getDataAesEncryptor(algorithm, key.size());

  std::string fileAad = properties_->fileAad();
  std::shared_ptr<Encryptor> encryptor =
      std::make_shared<Encryptor>(aesEncryptor, key, fileAad, "", pool_);
  if (metadata)
    columnMetadataMap_[columnPath] = encryptor;
  else
    columnDataMap_[columnPath] = encryptor;

  return encryptor;
}

int InternalFileEncryptor::mapKeyLenToEncryptorArrayIndex(int keyLen) {
  if (keyLen == 16)
    return 0;
  else if (keyLen == 24)
    return 1;
  else if (keyLen == 32)
    return 2;
  throw ParquetException("encryption key must be 16, 24 or 32 bytes in length");
}

encryption::AesEncryptor* InternalFileEncryptor::getMetaAesEncryptor(
    ParquetCipher::type algorithm,
    size_t keySize) {
  int keyLen = static_cast<int>(keySize);
  int index = mapKeyLenToEncryptorArrayIndex(keyLen);
  if (metaEncryptor_[index] == nullptr) {
    metaEncryptor_[index].reset(
        encryption::AesEncryptor::make(
            algorithm, keyLen, true, &allEncryptors_));
  }
  return metaEncryptor_[index].get();
}

encryption::AesEncryptor* InternalFileEncryptor::getDataAesEncryptor(
    ParquetCipher::type algorithm,
    size_t keySize) {
  int keyLen = static_cast<int>(keySize);
  int index = mapKeyLenToEncryptorArrayIndex(keyLen);
  if (dataEncryptor_[index] == nullptr) {
    dataEncryptor_[index].reset(
        encryption::AesEncryptor::make(
            algorithm, keyLen, false, &allEncryptors_));
  }
  return dataEncryptor_[index].get();
}

} // namespace facebook::velox::parquet::arrow
