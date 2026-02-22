/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/parquet/writer/arrow/Schema.h"

namespace facebook::velox::parquet::arrow {

namespace encryption {
class AesDecryptor;
class AesEncryptor;
} // namespace encryption

class FileDecryptionProperties;

class PARQUET_EXPORT Decryptor {
 public:
  Decryptor(
      std::shared_ptr<encryption::AesDecryptor> Decryptor,
      const std::string& key,
      const std::string& fileAad,
      const std::string& aad,
      ::arrow::MemoryPool* pool);

  const std::string& fileAad() const {
    return fileAad_;
  }
  void updateAad(const std::string& aad) {
    aad_ = aad;
  }
  ::arrow::MemoryPool* pool() {
    return pool_;
  }

  int ciphertextSizeDelta();
  int decrypt(const uint8_t* ciphertext, int ciphertextLen, uint8_t* plaintext);

 private:
  std::shared_ptr<encryption::AesDecryptor> aesDecryptor_;
  std::string key_;
  std::string fileAad_;
  std::string aad_;
  ::arrow::MemoryPool* pool_;
};

class InternalFileDecryptor {
 public:
  explicit InternalFileDecryptor(
      FileDecryptionProperties* properties,
      const std::string& fileAad,
      ParquetCipher::type algorithm,
      const std::string& footerKeyMetadata,
      ::arrow::MemoryPool* pool);

  std::string& fileAad() {
    return fileAad_;
  }

  std::string getFooterKey();

  ParquetCipher::type algorithm() {
    return algorithm_;
  }

  std::string& footerKeyMetadata() {
    return footerKeyMetadata_;
  }

  FileDecryptionProperties* properties() {
    return properties_;
  }

  void wipeOutDecryptionKeys();

  ::arrow::MemoryPool* pool() {
    return pool_;
  }

  std::shared_ptr<Decryptor> getFooterDecryptor();
  std::shared_ptr<Decryptor> getFooterDecryptorForColumnMeta(
      const std::string& aad = "");
  std::shared_ptr<Decryptor> getFooterDecryptorForColumnData(
      const std::string& aad = "");
  std::shared_ptr<Decryptor> getColumnMetaDecryptor(
      const std::string& ColumnPath,
      const std::string& columnKeyMetadata,
      const std::string& aad = "");
  std::shared_ptr<Decryptor> getColumnDataDecryptor(
      const std::string& ColumnPath,
      const std::string& columnKeyMetadata,
      const std::string& aad = "");

 private:
  FileDecryptionProperties* properties_;
  // Concatenation of aad_prefix (if exists) and aad_file_unique.
  std::string fileAad_;
  std::map<std::string, std::shared_ptr<Decryptor>> columnDataMap_;
  std::map<std::string, std::shared_ptr<Decryptor>> columnMetadataMap_;

  std::shared_ptr<Decryptor> footerMetadataDecryptor_;
  std::shared_ptr<Decryptor> footerDataDecryptor_;
  ParquetCipher::type algorithm_;
  std::string footerKeyMetadata_;
  // A weak reference to all decryptors that need to be wiped out when.
  // Decryption is finished.
  std::vector<std::weak_ptr<encryption::AesDecryptor>> allDecryptors_;

  ::arrow::MemoryPool* pool_;

  std::shared_ptr<Decryptor> getFooterDecryptor(
      const std::string& aad,
      bool metadata);
  std::shared_ptr<Decryptor> getColumnDecryptor(
      const std::string& ColumnPath,
      const std::string& columnKeyMetadata,
      const std::string& aad,
      bool metadata = false);
};

} // namespace facebook::velox::parquet::arrow
