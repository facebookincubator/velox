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

#include "velox/dwio/parquet/writer/arrow/Encryption.h"

#include <string.h>

#include <map>
#include <utility>

#include "arrow/util/utf8.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/writer/arrow/EncryptionInternal.h"

namespace facebook::velox::parquet::arrow {

// Integer key retriever.
void IntegerKeyIdRetriever::putKey(uint32_t keyId, const std::string& key) {
  keyMap_.insert({keyId, key});
}

std::string IntegerKeyIdRetriever::getKey(const std::string& keyMetadata) {
  uint32_t keyId;
  memcpy(reinterpret_cast<uint8_t*>(&keyId), keyMetadata.c_str(), 4);

  return keyMap_.at(keyId);
}

// String key retriever.
void StringKeyIdRetriever::putKey(
    const std::string& keyId,
    const std::string& key) {
  keyMap_.insert({keyId, key});
}

std::string StringKeyIdRetriever::getKey(const std::string& keyId) {
  return keyMap_.at(keyId);
}

ColumnEncryptionProperties::Builder* ColumnEncryptionProperties::Builder::key(
    std::string columnKey) {
  if (columnKey.empty())
    return this;

  VELOX_DCHECK(key_.empty());
  key_ = columnKey;
  return this;
}

ColumnEncryptionProperties::Builder*
ColumnEncryptionProperties::Builder::keyMetadata(
    const std::string& keyMetadata) {
  VELOX_DCHECK(!keyMetadata.empty());
  VELOX_DCHECK(keyMetadata_.empty());
  keyMetadata_ = keyMetadata;
  return this;
}

ColumnEncryptionProperties::Builder* ColumnEncryptionProperties::Builder::keyId(
    const std::string& keyId) {
  // Key_id is expected to be in UTF8 encoding.
  ::arrow::util::InitializeUTF8();
  const uint8_t* data = reinterpret_cast<const uint8_t*>(keyId.c_str());
  if (!::arrow::util::ValidateUTF8(data, keyId.size())) {
    throw ParquetException("key id should be in UTF8 encoding");
  }

  VELOX_DCHECK(!keyId.empty());
  this->keyMetadata(keyId);
  return this;
}

FileDecryptionProperties::Builder*
FileDecryptionProperties::Builder::columnKeys(
    const ColumnPathToDecryptionPropertiesMap& ColumnDecryptionProperties) {
  if (ColumnDecryptionProperties.size() == 0)
    return this;

  if (columnDecryptionProperties_.size() != 0)
    throw ParquetException("Column properties already set");

  for (const auto& element : ColumnDecryptionProperties) {
    if (element.second->isUtilized()) {
      throw ParquetException("Column properties utilized in another file");
    }
    element.second->setUtilized();
  }

  columnDecryptionProperties_ = ColumnDecryptionProperties;
  return this;
}

void FileDecryptionProperties::wipeOutDecryptionKeys() {
  footerKey_.clear();

  for (const auto& element : columnDecryptionProperties_) {
    element.second->wipeOutDecryptionKey();
  }
}

bool FileDecryptionProperties::isUtilized() {
  if (footerKey_.empty() && columnDecryptionProperties_.size() == 0 &&
      aadPrefix_.empty())
    return false;

  return utilized_;
}

std::shared_ptr<FileDecryptionProperties> FileDecryptionProperties::deepClone(
    std::string newAadPrefix) {
  std::string footerKeyCopy = footerKey_;
  ColumnPathToDecryptionPropertiesMap columnDecryptionPropertiesMapCopy;

  for (const auto& element : columnDecryptionProperties_) {
    columnDecryptionPropertiesMapCopy.insert(
        {element.second->columnPath(), element.second->deepClone()});
  }

  if (newAadPrefix.empty())
    newAadPrefix = aadPrefix_;
  return std::shared_ptr<FileDecryptionProperties>(new FileDecryptionProperties(
      footerKeyCopy,
      keyRetriever_,
      checkPlaintextFooterIntegrity_,
      newAadPrefix,
      aadPrefixVerifier_,
      columnDecryptionPropertiesMapCopy,
      plaintextFilesAllowed_));
}

FileDecryptionProperties::Builder* FileDecryptionProperties::Builder::footerKey(
    const std::string footerKey) {
  if (footerKey.empty()) {
    return this;
  }
  VELOX_DCHECK(footerKey_.empty());
  footerKey_ = footerKey;
  return this;
}

FileDecryptionProperties::Builder*
FileDecryptionProperties::Builder::keyRetriever(
    const std::shared_ptr<DecryptionKeyRetriever>& keyRetriever) {
  if (keyRetriever == nullptr)
    return this;

  VELOX_DCHECK_NULL(keyRetriever_);
  keyRetriever_ = keyRetriever;
  return this;
}

FileDecryptionProperties::Builder* FileDecryptionProperties::Builder::aadPrefix(
    const std::string& aadPrefix) {
  if (aadPrefix.empty()) {
    return this;
  }
  VELOX_DCHECK(aadPrefix_.empty());
  aadPrefix_ = aadPrefix;
  return this;
}

FileDecryptionProperties::Builder*
FileDecryptionProperties::Builder::aadPrefixVerifier(
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier) {
  if (aadPrefixVerifier == nullptr)
    return this;

  VELOX_DCHECK_NULL(aadPrefixVerifier_);
  aadPrefixVerifier_ = std::move(aadPrefixVerifier);
  return this;
}

ColumnDecryptionProperties::Builder* ColumnDecryptionProperties::Builder::key(
    const std::string& key) {
  if (key.empty())
    return this;

  VELOX_DCHECK(!key.empty());
  key_ = key;
  return this;
}

std::shared_ptr<ColumnDecryptionProperties>
ColumnDecryptionProperties::Builder::build() {
  return std::shared_ptr<ColumnDecryptionProperties>(
      new ColumnDecryptionProperties(columnPath_, key_));
}

void ColumnDecryptionProperties::wipeOutDecryptionKey() {
  key_.clear();
}

std::shared_ptr<ColumnDecryptionProperties>
ColumnDecryptionProperties::deepClone() {
  std::string keyCopy = key_;
  return std::shared_ptr<ColumnDecryptionProperties>(
      new ColumnDecryptionProperties(columnPath_, keyCopy));
}

FileEncryptionProperties::Builder*
FileEncryptionProperties::Builder::footerKeyMetadata(
    const std::string& footerKeyMetadata) {
  if (footerKeyMetadata.empty())
    return this;

  VELOX_DCHECK(footerKeyMetadata_.empty());
  footerKeyMetadata_ = footerKeyMetadata;
  return this;
}

FileEncryptionProperties::Builder*
FileEncryptionProperties::Builder::encryptedColumns(
    const ColumnPathToEncryptionPropertiesMap& encryptedColumns) {
  if (encryptedColumns.size() == 0)
    return this;

  if (encryptedColumns_.size() != 0)
    throw ParquetException("Column properties already set");

  for (const auto& element : encryptedColumns) {
    if (element.second->isUtilized()) {
      throw ParquetException("Column properties utilized in another file");
    }
    element.second->setUtilized();
  }
  encryptedColumns_ = encryptedColumns;
  return this;
}

void FileEncryptionProperties::wipeOutEncryptionKeys() {
  footerKey_.clear();
  for (const auto& element : encryptedColumns_) {
    element.second->wipeOutEncryptionKey();
  }
}

std::shared_ptr<FileEncryptionProperties> FileEncryptionProperties::deepClone(
    std::string newAadPrefix) {
  std::string footerKeyCopy = footerKey_;
  ColumnPathToEncryptionPropertiesMap encryptedColumnsMapCopy;

  for (const auto& element : encryptedColumns_) {
    encryptedColumnsMapCopy.insert(
        {element.second->columnPath(), element.second->deepClone()});
  }

  if (newAadPrefix.empty())
    newAadPrefix = aadPrefix_;
  return std::shared_ptr<FileEncryptionProperties>(new FileEncryptionProperties(
      algorithm_.algorithm,
      footerKeyCopy,
      footerKeyMetadata_,
      encryptedFooter_,
      newAadPrefix,
      storeAadPrefixInFile_,
      encryptedColumnsMapCopy));
}

FileEncryptionProperties::Builder* FileEncryptionProperties::Builder::aadPrefix(
    const std::string& aadPrefix) {
  if (aadPrefix.empty())
    return this;

  VELOX_DCHECK(aadPrefix_.empty());
  aadPrefix_ = aadPrefix;
  storeAadPrefixInFile_ = true;
  return this;
}

FileEncryptionProperties::Builder*
FileEncryptionProperties::Builder::disableAadPrefixStorage() {
  VELOX_DCHECK(!aadPrefix_.empty());

  storeAadPrefixInFile_ = false;
  return this;
}

ColumnEncryptionProperties::ColumnEncryptionProperties(
    bool encrypted,
    const std::string& ColumnPath,
    const std::string& key,
    const std::string& keyMetadata)
    : columnPath_(ColumnPath) {
  // Column encryption properties object (with a column key) can be used for.
  // Writing only one file. Upon completion of file writing, the encryption
  // keys. In the properties will be wiped out (set to 0 in memory).
  utilized_ = false;

  VELOX_DCHECK(!ColumnPath.empty());
  if (!encrypted) {
    VELOX_DCHECK(key.empty() && keyMetadata.empty());
  }

  if (!key.empty()) {
    VELOX_DCHECK(
        key.length() == 16 || key.length() == 24 || key.length() == 32);
  }

  encryptedWithFooterKey_ = (encrypted && key.empty());
  if (encryptedWithFooterKey_) {
    VELOX_DCHECK(keyMetadata.empty());
  }

  encrypted_ = encrypted;
  keyMetadata_ = keyMetadata;
  key_ = key;
}

ColumnDecryptionProperties::ColumnDecryptionProperties(
    const std::string& ColumnPath,
    const std::string& key)
    : columnPath_(ColumnPath) {
  utilized_ = false;
  VELOX_DCHECK(!ColumnPath.empty());

  if (!key.empty()) {
    VELOX_DCHECK(
        key.length() == 16 || key.length() == 24 || key.length() == 32);
  }

  key_ = key;
}

std::string FileDecryptionProperties::columnKey(
    const std::string& ColumnPath) const {
  if (columnDecryptionProperties_.find(ColumnPath) !=
      columnDecryptionProperties_.end()) {
    auto columnProp = columnDecryptionProperties_.at(ColumnPath);
    if (columnProp != nullptr) {
      return columnProp->key();
    }
  }
  return emptyString_;
}

FileDecryptionProperties::FileDecryptionProperties(
    const std::string& footerKey,
    std::shared_ptr<DecryptionKeyRetriever> keyRetriever,
    bool checkPlaintextFooterIntegrity,
    const std::string& aadPrefix,
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier,
    const ColumnPathToDecryptionPropertiesMap& ColumnDecryptionProperties,
    bool plaintextFilesAllowed) {
  VELOX_DCHECK(
      !footerKey.empty() || nullptr != keyRetriever ||
      0 != ColumnDecryptionProperties.size());

  if (!footerKey.empty()) {
    VELOX_DCHECK(
        footerKey.length() == 16 || footerKey.length() == 24 ||
        footerKey.length() == 32);
  }
  if (footerKey.empty() && checkPlaintextFooterIntegrity) {
    VELOX_DCHECK_NOT_NULL(keyRetriever);
  }
  aadPrefixVerifier_ = std::move(aadPrefixVerifier);
  footerKey_ = footerKey;
  checkPlaintextFooterIntegrity_ = checkPlaintextFooterIntegrity;
  keyRetriever_ = std::move(keyRetriever);
  aadPrefix_ = aadPrefix;
  columnDecryptionProperties_ = ColumnDecryptionProperties;
  plaintextFilesAllowed_ = plaintextFilesAllowed;
  utilized_ = false;
}

FileEncryptionProperties::Builder*
FileEncryptionProperties::Builder::footerKeyId(const std::string& keyId) {
  // Key_id is expected to be in UTF8 encoding.
  ::arrow::util::InitializeUTF8();
  const uint8_t* data = reinterpret_cast<const uint8_t*>(keyId.c_str());
  if (!::arrow::util::ValidateUTF8(data, keyId.size())) {
    throw ParquetException("footer key id should be in UTF8 encoding");
  }

  if (keyId.empty()) {
    return this;
  }

  return footerKeyMetadata(keyId);
}

std::shared_ptr<ColumnEncryptionProperties>
FileEncryptionProperties::columnEncryptionProperties(
    const std::string& columnPath) {
  if (encryptedColumns_.empty()) {
    auto builder =
        std::make_shared<ColumnEncryptionProperties::Builder>(columnPath);
    return builder->build();
  }
  if (encryptedColumns_.find(columnPath) != encryptedColumns_.end()) {
    return encryptedColumns_[columnPath];
  }

  return nullptr;
}

FileEncryptionProperties::FileEncryptionProperties(
    ParquetCipher::type cipher,
    const std::string& footerKey,
    const std::string& footerKeyMetadata,
    bool encryptedFooter,
    const std::string& aadPrefix,
    bool storeAadPrefixInFile,
    const ColumnPathToEncryptionPropertiesMap& encryptedColumns)
    : footerKey_(footerKey),
      footerKeyMetadata_(footerKeyMetadata),
      encryptedFooter_(encryptedFooter),
      aadPrefix_(aadPrefix),
      storeAadPrefixInFile_(storeAadPrefixInFile),
      encryptedColumns_(encryptedColumns) {
  // File encryption properties object can be used for writing only one file.
  // Upon completion of file writing, the encryption keys in the properties
  // will. Be wiped out (set to 0 in memory).
  utilized_ = false;

  VELOX_DCHECK(!footerKey.empty());
  // Footer_key must be either 16, 24 or 32 bytes.
  VELOX_DCHECK(
      footerKey.length() == 16 || footerKey.length() == 24 ||
      footerKey.length() == 32);

  uint8_t aadFileUnique[kAadFileUniqueLength];
  encryption::randBytes(aadFileUnique, kAadFileUniqueLength);
  std::string aadFileUniqueStr(
      reinterpret_cast<char const*>(aadFileUnique), kAadFileUniqueLength);

  bool supplyAadPrefix = false;
  if (aadPrefix.empty()) {
    fileAad_ = aadFileUniqueStr;
  } else {
    fileAad_ = aadPrefix + aadFileUniqueStr;
    if (!storeAadPrefixInFile)
      supplyAadPrefix = true;
  }
  algorithm_.algorithm = cipher;
  algorithm_.aad.aadFileUnique = aadFileUniqueStr;
  algorithm_.aad.supplyAadPrefix = supplyAadPrefix;
  if (!aadPrefix.empty() && storeAadPrefixInFile) {
    algorithm_.aad.aadPrefix = aadPrefix;
  }
}

} // namespace facebook::velox::parquet::arrow
