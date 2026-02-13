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

#include "velox/dwio/parquet/writer/arrow/FileDecryptorInternal.h"
#include "velox/dwio/parquet/writer/arrow/Encryption.h"
#include "velox/dwio/parquet/writer/arrow/EncryptionInternal.h"

namespace facebook::velox::parquet::arrow {

// Decryptor.
Decryptor::Decryptor(
    std::shared_ptr<encryption::AesDecryptor> aesDecryptor,
    const std::string& key,
    const std::string& fileAad,
    const std::string& aad,
    ::arrow::MemoryPool* pool)
    : aesDecryptor_(aesDecryptor),
      key_(key),
      fileAad_(fileAad),
      aad_(aad),
      pool_(pool) {}

int Decryptor::ciphertextSizeDelta() {
  return aesDecryptor_->ciphertextSizeDelta();
}

int Decryptor::decrypt(
    const uint8_t* ciphertext,
    int ciphertextLen,
    uint8_t* plaintext) {
  return aesDecryptor_->decrypt(
      ciphertext,
      ciphertextLen,
      str2bytes(key_),
      static_cast<int>(key_.size()),
      str2bytes(aad_),
      static_cast<int>(aad_.size()),
      plaintext);
}

// InternalFileDecryptor.
InternalFileDecryptor::InternalFileDecryptor(
    FileDecryptionProperties* properties,
    const std::string& fileAad,
    ParquetCipher::type algorithm,
    const std::string& footerKeyMetadata,
    ::arrow::MemoryPool* pool)
    : properties_(properties),
      fileAad_(fileAad),
      algorithm_(algorithm),
      footerKeyMetadata_(footerKeyMetadata),
      pool_(pool) {
  if (properties_->isUtilized()) {
    throw ParquetException(
        "Re-using decryption properties with explicit keys for another file");
  }
  properties_->setUtilized();
}

void InternalFileDecryptor::wipeOutDecryptionKeys() {
  properties_->wipeOutDecryptionKeys();
  for (auto const& i : allDecryptors_) {
    if (auto aesDecryptor = i.lock()) {
      aesDecryptor->wipeOut();
    }
  }
}

std::string InternalFileDecryptor::getFooterKey() {
  std::string footerKey = properties_->footerKey();
  // Ignore footer key metadata if footer key is explicitly set via API.
  if (footerKey.empty()) {
    if (footerKeyMetadata_.empty())
      throw ParquetException("No footer key or key metadata");
    if (properties_->keyRetriever() == nullptr)
      throw ParquetException("No footer key or key retriever");
    try {
      footerKey = properties_->keyRetriever()->getKey(footerKeyMetadata_);
    } catch (KeyAccessDeniedException& e) {
      std::stringstream ss;
      ss << "Footer key: access denied " << e.what() << "\n";
      throw ParquetException(ss.str());
    }
  }
  if (footerKey.empty()) {
    throw ParquetException(
        "Footer key unavailable. Could not verify "
        "plaintext footer metadata");
  }
  return footerKey;
}

std::shared_ptr<Decryptor> InternalFileDecryptor::getFooterDecryptor() {
  std::string aad = encryption::createFooterAad(fileAad_);
  return getFooterDecryptor(aad, true);
}

std::shared_ptr<Decryptor>
InternalFileDecryptor::getFooterDecryptorForColumnMeta(const std::string& aad) {
  return getFooterDecryptor(aad, true);
}

std::shared_ptr<Decryptor>
InternalFileDecryptor::getFooterDecryptorForColumnData(const std::string& aad) {
  return getFooterDecryptor(aad, false);
}

std::shared_ptr<Decryptor> InternalFileDecryptor::getFooterDecryptor(
    const std::string& aad,
    bool metadata) {
  if (metadata) {
    if (footerMetadataDecryptor_ != nullptr)
      return footerMetadataDecryptor_;
  } else {
    if (footerDataDecryptor_ != nullptr)
      return footerDataDecryptor_;
  }

  std::string footerKey = properties_->footerKey();
  if (footerKey.empty()) {
    if (footerKeyMetadata_.empty())
      throw ParquetException("No footer key or key metadata");
    if (properties_->keyRetriever() == nullptr)
      throw ParquetException("No footer key or key retriever");
    try {
      footerKey = properties_->keyRetriever()->getKey(footerKeyMetadata_);
    } catch (KeyAccessDeniedException& e) {
      std::stringstream ss;
      ss << "Footer key: access denied " << e.what() << "\n";
      throw ParquetException(ss.str());
    }
  }
  if (footerKey.empty()) {
    throw ParquetException(
        "Invalid footer encryption key. "
        "Could not parse footer metadata");
  }

  // Create both data and metadata decryptors to avoid redundant retrieval of.
  // Key from the key_retriever.
  int keyLen = static_cast<int>(footerKey.size());
  auto aesMetadataDecryptor =
      encryption::AesDecryptor::make(algorithm_, keyLen, true, &allDecryptors_);
  auto aesDataDecryptor = encryption::AesDecryptor::make(
      algorithm_, keyLen, false, &allDecryptors_);

  footerMetadataDecryptor_ = std::make_shared<Decryptor>(
      aesMetadataDecryptor, footerKey, fileAad_, aad, pool_);
  footerDataDecryptor_ = std::make_shared<Decryptor>(
      aesDataDecryptor, footerKey, fileAad_, aad, pool_);

  if (metadata)
    return footerMetadataDecryptor_;
  return footerDataDecryptor_;
}

std::shared_ptr<Decryptor> InternalFileDecryptor::getColumnMetaDecryptor(
    const std::string& ColumnPath,
    const std::string& columnKeyMetadata,
    const std::string& aad) {
  return getColumnDecryptor(ColumnPath, columnKeyMetadata, aad, true);
}

std::shared_ptr<Decryptor> InternalFileDecryptor::getColumnDataDecryptor(
    const std::string& ColumnPath,
    const std::string& columnKeyMetadata,
    const std::string& aad) {
  return getColumnDecryptor(ColumnPath, columnKeyMetadata, aad, false);
}

std::shared_ptr<Decryptor> InternalFileDecryptor::getColumnDecryptor(
    const std::string& ColumnPath,
    const std::string& columnKeyMetadata,
    const std::string& aad,
    bool metadata) {
  std::string columnKey;
  // First look if we already got the decryptor from before.
  if (metadata) {
    if (columnMetadataMap_.find(ColumnPath) != columnMetadataMap_.end()) {
      auto res(columnMetadataMap_.at(ColumnPath));
      res->updateAad(aad);
      return res;
    }
  } else {
    if (columnDataMap_.find(ColumnPath) != columnDataMap_.end()) {
      auto res(columnDataMap_.at(ColumnPath));
      res->updateAad(aad);
      return res;
    }
  }

  columnKey = properties_->columnKey(ColumnPath);
  // No explicit column key given via API. Retrieve via key metadata.
  if (columnKey.empty() && !columnKeyMetadata.empty() &&
      properties_->keyRetriever() != nullptr) {
    try {
      columnKey = properties_->keyRetriever()->getKey(columnKeyMetadata);
    } catch (KeyAccessDeniedException& e) {
      std::stringstream ss;
      ss << "HiddenColumnException, path=" + ColumnPath + " " << e.what()
         << "\n";
      throw HiddenColumnException(ss.str());
    }
  }
  if (columnKey.empty()) {
    throw HiddenColumnException("HiddenColumnException, path=" + ColumnPath);
  }

  // Create both data and metadata decryptors to avoid redundant retrieval of.
  // Key using the key_retriever.
  int keyLen = static_cast<int>(columnKey.size());
  auto aesMetadataDecryptor =
      encryption::AesDecryptor::make(algorithm_, keyLen, true, &allDecryptors_);
  auto aesDataDecryptor = encryption::AesDecryptor::make(
      algorithm_, keyLen, false, &allDecryptors_);

  columnMetadataMap_[ColumnPath] = std::make_shared<Decryptor>(
      aesMetadataDecryptor, columnKey, fileAad_, aad, pool_);
  columnDataMap_[ColumnPath] = std::make_shared<Decryptor>(
      aesDataDecryptor, columnKey, fileAad_, aad, pool_);

  if (metadata)
    return columnMetadataMap_[ColumnPath];
  return columnDataMap_[ColumnPath];
}

} // namespace facebook::velox::parquet::arrow
