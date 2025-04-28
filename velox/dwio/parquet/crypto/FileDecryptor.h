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

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "velox/dwio/parquet/crypto/AesEncryption.h"
#include "velox/dwio/parquet/crypto/ColumnPath.h"
#include "velox/dwio/parquet/crypto/FileDecryptionProperties.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

class ColumnDecryptionSetup;
using ColumnPathToDecryptionSetupMap = std::map<std::string, std::shared_ptr<ColumnDecryptionSetup>>;

class Decryptor {
 public:
  Decryptor(std::shared_ptr<AesDecryptor> decryptor, std::string  key,
            std::string  fileAad);

  const std::string& fileAad() const { return fileAad_; }
//  void updateAad(const std::string& aad) { aad_ = aad; }

  int plaintextLength(int ciphertextLen) const;
  int ciphertextLength(int plaintextLen) const;
  int decrypt(const uint8_t* ciphertext, int ciphertextLen,
              uint8_t* plaintext, int plaintextLen, std::string_view aad);

  std::shared_ptr<AesDecryptor> getAesDecryptor() {return aesDecryptor_;}

 private:
  std::shared_ptr<AesDecryptor> aesDecryptor_;
  std::string key_;
  std::string fileAad_;
};

class ColumnDecryptionSetup {
 public:
  explicit ColumnDecryptionSetup(ColumnPath& columnPath, bool encrypted, bool keyAvailable,
                                         std::shared_ptr<Decryptor> dataDecryptor,
                                         std::shared_ptr<Decryptor> metadataDecryptor,
                                         int columnOrdinal,
                                         std::string_view savedException) :
        columnPath_(columnPath), encrypted_(encrypted), keyAvailable_(keyAvailable),
        dataDecryptor_(std::move(dataDecryptor)), metadataDecryptor_(std::move(metadataDecryptor)),
        columnOrdinal_(columnOrdinal),
        savedException_(savedException) {}

  ColumnPath getColumnPath() {return columnPath_;}
  bool isEncrypted() {return encrypted_;}
  std::shared_ptr<Decryptor> getDataDecryptor () {return dataDecryptor_;}
  std::shared_ptr<Decryptor> getMetadataDecryptor () {return metadataDecryptor_;}
  int getColumnOrdinal() {return columnOrdinal_;}
  bool isKeyAvailable() {return keyAvailable_;}
  std::string savedException() {return savedException_;}

 private:
  ColumnPath columnPath_;
  bool encrypted_;
  bool keyAvailable_;
  std::shared_ptr<Decryptor> dataDecryptor_;
  std::shared_ptr<Decryptor> metadataDecryptor_;
  int columnOrdinal_;
  std::string savedException_;
};

class FileDecryptor {
 public:
  FileDecryptor(FileDecryptionProperties* properties,
                        std::string  fileAad,
                        ParquetCipher::type algorithm,
                        std::string user);

  ParquetCipher::type algorithm() { return algorithm_; }

  FileDecryptionProperties* properties() { return properties_; }

  std::string& user() { return user_; }

  std::shared_ptr<ColumnDecryptionSetup> setColumnCryptoMetadata(
      ColumnPath& columnPath,
      bool encrypted,
      std::string& keyMetadata,
      int columnOrdinal);

  std::shared_ptr<ColumnDecryptionSetup> getColumnCryptoMetadata(const std::string& columnPath) {
      const auto it = columnPathToDecryptionSetupMap_.find(columnPath);
      if (it == columnPathToDecryptionSetupMap_.end()) {
        return nullptr;
      }
      return it->second;
  }

  static std::string handleAadPrefix(FileDecryptionProperties* fileDecryptionProperties, thrift::EncryptionAlgorithm& encryptionAlgorithm);
  static ParquetCipher::type getEncryptionAlgorithm(thrift::EncryptionAlgorithm& encryptionAlgorithm);

 private:
  std::shared_ptr<Decryptor> getColumnMetaDecryptor(const std::string& column_key);
  std::shared_ptr<Decryptor> getColumnDataDecryptor(const std::string& column_key);
  std::shared_ptr<Decryptor> getColumnDecryptor(const std::string& columnKey, bool metadata = false);

  FileDecryptionProperties* properties_;
  std::string fileAad_;
  ParquetCipher::type algorithm_;
  ColumnPathToDecryptionSetupMap columnPathToDecryptionSetupMap_;
  std::string user_;
};

}
