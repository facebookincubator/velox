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

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "velox/dwio/parquet/writer/arrow/Encryption.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"

namespace facebook::velox::parquet::arrow {

namespace encryption {
class AesEncryptor;
} // namespace encryption

class FileEncryptionProperties;
class ColumnEncryptionProperties;

class PARQUET_EXPORT Encryptor {
 public:
  Encryptor(
      encryption::AesEncryptor* aesEncryptor,
      const std::string& key,
      const std::string& fileAad,
      const std::string& aad,
      ::arrow::MemoryPool* pool);
  const std::string& fileAad() {
    return fileAad_;
  }
  void updateAad(const std::string& aad) {
    aad_ = aad;
  }
  ::arrow::MemoryPool* pool() {
    return pool_;
  }

  int ciphertextSizeDelta();
  int encrypt(const uint8_t* plaintext, int plaintextLen, uint8_t* ciphertext);

  bool encryptColumnMetaData(
      bool encryptedFooter,
      const std::shared_ptr<ColumnEncryptionProperties>&
          ColumnEncryptionProperties) {
    // If column is not encrypted then do not encrypt the column metadata.
    if (!ColumnEncryptionProperties ||
        !ColumnEncryptionProperties->isEncrypted())
      return false;
    // If plaintext footer then encrypt the column metadata.
    if (!encryptedFooter)
      return true;
    // If column is not encrypted with footer key then encrypt the column.
    // Metadata.
    return !ColumnEncryptionProperties->isEncryptedWithFooterKey();
  }

 private:
  encryption::AesEncryptor* aesEncryptor_;
  std::string key_;
  std::string fileAad_;
  std::string aad_;
  ::arrow::MemoryPool* pool_;
};

class InternalFileEncryptor {
 public:
  explicit InternalFileEncryptor(
      FileEncryptionProperties* properties,
      ::arrow::MemoryPool* pool);

  std::shared_ptr<Encryptor> getFooterEncryptor();
  std::shared_ptr<Encryptor> getFooterSigningEncryptor();
  std::shared_ptr<Encryptor> getColumnMetaEncryptor(
      const std::string& ColumnPath);
  std::shared_ptr<Encryptor> getColumnDataEncryptor(
      const std::string& ColumnPath);
  void wipeOutEncryptionKeys();

 private:
  FileEncryptionProperties* properties_;

  std::map<std::string, std::shared_ptr<Encryptor>> columnDataMap_;
  std::map<std::string, std::shared_ptr<Encryptor>> columnMetadataMap_;

  std::shared_ptr<Encryptor> footerSigningEncryptor_;
  std::shared_ptr<Encryptor> footerEncryptor_;

  std::vector<encryption::AesEncryptor*> allEncryptors_;

  // Key must be 16, 24 or 32 bytes in length. Thus there could be up to three.
  // Types of meta_encryptors and data_encryptors.
  std::unique_ptr<encryption::AesEncryptor> metaEncryptor_[3];
  std::unique_ptr<encryption::AesEncryptor> dataEncryptor_[3];

  ::arrow::MemoryPool* pool_;

  std::shared_ptr<Encryptor> getColumnEncryptor(
      const std::string& ColumnPath,
      bool metadata);

  encryption::AesEncryptor* getMetaAesEncryptor(
      ParquetCipher::type algorithm,
      size_t keyLen);
  encryption::AesEncryptor* getDataAesEncryptor(
      ParquetCipher::type algorithm,
      size_t keyLen);

  int mapKeyLenToEncryptorArrayIndex(int keyLen);
};

} // namespace facebook::velox::parquet::arrow
