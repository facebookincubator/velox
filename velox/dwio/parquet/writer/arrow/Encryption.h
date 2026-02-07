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
#include <utility>

#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

namespace facebook::velox::parquet::arrow {

static constexpr ParquetCipher::type kDefaultEncryptionAlgorithm =
    ParquetCipher::kAesGcmV1;
static constexpr int32_t kMaximalAadMetadataLength = 256;
static constexpr bool kDefaultEncryptedFooter = true;
static constexpr bool kDefaultCheckSignature = true;
static constexpr bool kDefaultAllowPlaintextFiles = false;
static constexpr int32_t kAadFileUniqueLength = 8;

class ColumnDecryptionProperties;
using ColumnPathToDecryptionPropertiesMap =
    std::map<std::string, std::shared_ptr<ColumnDecryptionProperties>>;

class ColumnEncryptionProperties;
using ColumnPathToEncryptionPropertiesMap =
    std::map<std::string, std::shared_ptr<ColumnEncryptionProperties>>;

class PARQUET_EXPORT DecryptionKeyRetriever {
 public:
  virtual std::string getKey(const std::string& keyMetadata) = 0;
  virtual ~DecryptionKeyRetriever() {}
};

/// Simple integer key retriever.
class PARQUET_EXPORT IntegerKeyIdRetriever : public DecryptionKeyRetriever {
 public:
  void putKey(uint32_t keyId, const std::string& key);
  std::string getKey(const std::string& keyMetadata) override;

 private:
  std::map<uint32_t, std::string> keyMap_;
};

// Simple string key retriever.
class PARQUET_EXPORT StringKeyIdRetriever : public DecryptionKeyRetriever {
 public:
  void putKey(const std::string& keyId, const std::string& key);
  std::string getKey(const std::string& keyMetadata) override;

 private:
  std::map<std::string, std::string> keyMap_;
};

class PARQUET_EXPORT HiddenColumnException : public ParquetException {
 public:
  explicit HiddenColumnException(const std::string& columnPath)
      : ParquetException(columnPath.c_str()) {}
};

class PARQUET_EXPORT KeyAccessDeniedException : public ParquetException {
 public:
  explicit KeyAccessDeniedException(const std::string& columnPath)
      : ParquetException(columnPath.c_str()) {}
};

inline const uint8_t* str2bytes(const std::string& str) {
  if (str.empty())
    return NULLPTR;

  char* cbytes = const_cast<char*>(str.c_str());
  return reinterpret_cast<const uint8_t*>(cbytes);
}

class PARQUET_EXPORT ColumnEncryptionProperties {
 public:
  class PARQUET_EXPORT Builder {
   public:
    /// Convenience builder for encrypted columns.
    explicit Builder(const std::string& name) : Builder(name, true) {}

    /// Convenience builder for encrypted columns.
    explicit Builder(const std::shared_ptr<schema::ColumnPath>& path)
        : Builder(path->toDotString(), true) {}

    /// Set a column-specific key.
    /// If key is not set on an encrypted column, the column will
    /// be encrypted with the footer key.
    /// KeyBytes Key length must be either 16, 24 or 32 bytes.
    /// The key is cloned, and will be wiped out (array values set to 0) upon
    /// completion of file writing. Caller is responsible for wiping out the.
    /// input key array.
    Builder* key(std::string columnKey);

    /// Set a key retrieval metadata.
    /// Use either key_metadata() or key_id(), not both.
    Builder* keyMetadata(const std::string& keyMetadata);

    /// A convenience function to set key metadata using a string id.
    /// Set a key retrieval metadata (converted from String).
    /// Use either key_metadata() or key_id(), not both.
    /// key_id will be converted to metadata (UTF-8 array).
    Builder* keyId(const std::string& keyId);

    std::shared_ptr<ColumnEncryptionProperties> build() {
      return std::shared_ptr<ColumnEncryptionProperties>(
          new ColumnEncryptionProperties(
              encrypted_, columnPath_, key_, keyMetadata_));
    }

   private:
    const std::string columnPath_;
    bool encrypted_;
    std::string key_;
    std::string keyMetadata_;

    Builder(const std::string path, bool encrypted)
        : columnPath_(path), encrypted_(encrypted) {}
  };

  std::string columnPath() const {
    return columnPath_;
  }
  bool isEncrypted() const {
    return encrypted_;
  }
  bool isEncryptedWithFooterKey() const {
    return encryptedWithFooterKey_;
  }
  std::string key() const {
    return key_;
  }
  std::string keyMetadata() const {
    return keyMetadata_;
  }

  /// Upon completion of file writing, the encryption key
  /// will be wiped out.
  void wipeOutEncryptionKey() {
    key_.clear();
  }

  bool isUtilized() {
    if (key_.empty())
      return false; // can re-use column properties without encryption keys
    return utilized_;
  }

  /// ColumnEncryptionProperties object can be used for writing one file only.
  /// Mark ColumnEncryptionProperties as utilized once it is used in
  /// FileEncryptionProperties as the encryption key will be wiped out upon
  /// completion of file writing.
  void setUtilized() {
    utilized_ = true;
  }

  std::shared_ptr<ColumnEncryptionProperties> deepClone() {
    std::string keyCopy = key_;
    return std::shared_ptr<ColumnEncryptionProperties>(
        new ColumnEncryptionProperties(
            encrypted_, columnPath_, keyCopy, keyMetadata_));
  }

  ColumnEncryptionProperties() = default;
  ColumnEncryptionProperties(const ColumnEncryptionProperties& other) = default;
  ColumnEncryptionProperties(ColumnEncryptionProperties&& other) = default;

 private:
  const std::string columnPath_;
  bool encrypted_;
  bool encryptedWithFooterKey_;
  std::string key_;
  std::string keyMetadata_;
  bool utilized_;
  explicit ColumnEncryptionProperties(
      bool encrypted,
      const std::string& columnPath,
      const std::string& key,
      const std::string& keyMetadata);
};

class PARQUET_EXPORT ColumnDecryptionProperties {
 public:
  class PARQUET_EXPORT Builder {
   public:
    explicit Builder(const std::string& name) : columnPath_(name) {}

    explicit Builder(const std::shared_ptr<schema::ColumnPath>& path)
        : Builder(path->toDotString()) {}

    /// Set an explicit column key. If applied on a file that contains
    /// key metadata for this column the metadata will be ignored,
    /// and the column will be decrypted with this key.
    /// Key length must be either 16, 24 or 32 bytes.
    Builder* key(const std::string& key);

    std::shared_ptr<ColumnDecryptionProperties> build();

   private:
    const std::string columnPath_;
    std::string key_;
  };

  ColumnDecryptionProperties() = default;
  ColumnDecryptionProperties(const ColumnDecryptionProperties& other) = default;
  ColumnDecryptionProperties(ColumnDecryptionProperties&& other) = default;

  std::string columnPath() const {
    return columnPath_;
  }
  std::string key() const {
    return key_;
  }
  bool isUtilized() {
    return utilized_;
  }

  /// ColumnDecryptionProperties object can be used for reading one file only.
  /// Mark ColumnDecryptionProperties as utilized once it is used in
  /// FileDecryptionProperties as the encryption key will be wiped out upon
  /// completion of file reading.
  void setUtilized() {
    utilized_ = true;
  }

  /// Upon completion of file reading, the encryption key
  /// will be wiped out.
  void wipeOutDecryptionKey();

  std::shared_ptr<ColumnDecryptionProperties> deepClone();

 private:
  const std::string columnPath_;
  std::string key_;
  bool utilized_;

  /// This class is only required for setting explicit column decryption keys -.
  /// To override key retriever (or to provide keys when key metadata and/or.
  /// key retriever are not available)
  explicit ColumnDecryptionProperties(
      const std::string& columnPath,
      const std::string& key);
};

class PARQUET_EXPORT AADPrefixVerifier {
 public:
  /// Verifies identity (AAD Prefix) of individual file,
  /// or of file collection in a data set.
  /// Throws exception if an AAD prefix is wrong.
  /// In a data set, AAD Prefixes should be collected,
  /// and then checked for missing files.
  virtual void verify(const std::string& aadPrefix) = 0;
  virtual ~AADPrefixVerifier() {}
};

class PARQUET_EXPORT FileDecryptionProperties {
 public:
  class PARQUET_EXPORT Builder {
   public:
    Builder() {
      checkPlaintextFooterIntegrity_ = kDefaultCheckSignature;
      plaintextFilesAllowed_ = kDefaultAllowPlaintextFiles;
    }

    /// Set an explicit footer key. If applied on a file that contains
    /// footer key metadata the metadata will be ignored, the footer
    /// will be decrypted/verified with this key.
    /// If explicit key is not set, footer key will be fetched from
    /// key retriever.
    /// With explicit keys or AAD prefix, new encryption properties object must
    /// be created for each encrypted file. Explicit encryption keys (footer
    /// and column) are cloned. Upon completion of file reading, the cloned
    /// encryption keys in the properties will be wiped out (array values set
    /// to 0). Caller is responsible for wiping out the input key array. param
    /// footerKey Key length must be either 16, 24 or 32 bytes.
    Builder* footerKey(const std::string footerKey);

    /// Set explicit column keys (decryption properties).
    /// Its also possible to set a key retriever on this property object.
    /// Upon file decryption, availability of explicit keys is checked before.
    /// invocation of the retriever callback.
    /// If an explicit key is available for a footer or a column,
    /// its key metadata will be ignored.
    Builder* columnKeys(
        const ColumnPathToDecryptionPropertiesMap& columnDecryptionProperties);

    /// Set a key retrieval callback. It is also possible to
    /// set explicit footer or column keys on this file property object.
    /// Upon file decryption, availability of explicit keys is checked before.
    /// invocation of the retriever callback.
    /// If an explicit key is available for a footer or a column,
    /// its key metadata will be ignored.
    Builder* keyRetriever(
        const std::shared_ptr<DecryptionKeyRetriever>& keyRetriever);

    /// Skip integrity verification of plaintext footers.
    /// If not called, integrity of plaintext footers will be checked in.
    /// Runtime, and an exception will be thrown in the following situations:
    /// - Footer signing key is not available.
    /// (not passed, or not found by key retriever)..
    /// - Footer content and signature don't match.
    Builder* disableFooterSignatureVerification() {
      checkPlaintextFooterIntegrity_ = false;
      return this;
    }

    /// Explicitly supply the file AAD prefix.
    /// This is mandatory when a prefix is used for file encryption, but not
    /// stored in file. If AAD prefix is stored in file, it will be compared to
    /// the explicitly supplied value and an exception will be thrown if they
    /// differ.
    Builder* aadPrefix(const std::string& aadPrefix);

    /// Set callback for verification of AAD Prefixes stored in file.
    Builder* aadPrefixVerifier(
        std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier);

    /// By default, reading plaintext (unencrypted) files is not
    /// allowed when using a decryptor.
    /// - in order to detect files that were not encrypted by mistake.
    /// However, the default behavior can be overridden by calling this method.
    /// The caller should then use a different method to ensure encryption
    /// of files with sensitive data.
    Builder* plaintextFilesAllowed() {
      plaintextFilesAllowed_ = true;
      return this;
    }

    std::shared_ptr<FileDecryptionProperties> build() {
      return std::shared_ptr<FileDecryptionProperties>(
          new FileDecryptionProperties(
              footerKey_,
              keyRetriever_,
              checkPlaintextFooterIntegrity_,
              aadPrefix_,
              aadPrefixVerifier_,
              columnDecryptionProperties_,
              plaintextFilesAllowed_));
    }

   private:
    std::string footerKey_;
    std::string aadPrefix_;
    std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier_;
    ColumnPathToDecryptionPropertiesMap columnDecryptionProperties_;

    std::shared_ptr<DecryptionKeyRetriever> keyRetriever_;
    bool checkPlaintextFooterIntegrity_;
    bool plaintextFilesAllowed_;
  };

  std::string columnKey(const std::string& columnPath) const;

  std::string footerKey() const {
    return footerKey_;
  }

  std::string aadPrefix() const {
    return aadPrefix_;
  }

  const std::shared_ptr<DecryptionKeyRetriever>& keyRetriever() const {
    return keyRetriever_;
  }

  bool checkPlaintextFooterIntegrity() const {
    return checkPlaintextFooterIntegrity_;
  }

  bool plaintextFilesAllowed() const {
    return plaintextFilesAllowed_;
  }

  const std::shared_ptr<AADPrefixVerifier>& aadPrefixVerifier() const {
    return aadPrefixVerifier_;
  }

  /// Upon completion of file reading, the encryption keys in the properties
  /// will be wiped out (array values set to 0).
  void wipeOutDecryptionKeys();

  bool isUtilized();

  /// FileDecryptionProperties object can be used for reading one file only.
  /// Mark FileDecryptionProperties as utilized once it is used to read a file
  /// as the encryption keys will be wiped out upon completion of file reading.
  void setUtilized() {
    utilized_ = true;
  }

  /// FileDecryptionProperties object can be used for reading one file only
  /// (unless this object keeps the keyRetrieval callback only, and no explicit
  /// keys or aadPrefix).
  /// At the end, keys are wiped out in the memory.
  /// This method allows cloning identical properties for another file,
  /// with an option to update the aadPrefix (if newAadPrefix is null,
  /// aadPrefix will be cloned too).
  std::shared_ptr<FileDecryptionProperties> deepClone(
      std::string newAadPrefix = "");

 private:
  std::string footerKey_;
  std::string aadPrefix_;
  std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier_;

  const std::string emptyString_ = "";
  ColumnPathToDecryptionPropertiesMap columnDecryptionProperties_;

  std::shared_ptr<DecryptionKeyRetriever> keyRetriever_;
  bool checkPlaintextFooterIntegrity_;
  bool plaintextFilesAllowed_;
  bool utilized_;

  FileDecryptionProperties(
      const std::string& footerKey,
      std::shared_ptr<DecryptionKeyRetriever> keyRetriever,
      bool checkPlaintextFooterIntegrity,
      const std::string& aadPrefix,
      std::shared_ptr<AADPrefixVerifier> aadPrefixVerifier,
      const ColumnPathToDecryptionPropertiesMap& columnDecryptionProperties,
      bool plaintextFilesAllowed);
};

class PARQUET_EXPORT FileEncryptionProperties {
 public:
  class PARQUET_EXPORT Builder {
   public:
    explicit Builder(const std::string& footerKey)
        : parquetCipher_(kDefaultEncryptionAlgorithm),
          encryptedFooter_(kDefaultEncryptedFooter) {
      footerKey_ = footerKey;
      storeAadPrefixInFile_ = false;
    }

    /// Create files with plaintext footer.
    /// If not called, the files will be created with encrypted footer.
    /// (default).
    Builder* setPlaintextFooter() {
      encryptedFooter_ = false;
      return this;
    }

    /// Set encryption algorithm.
    /// If not called, files will be encrypted with AES_GCM_V1 (default).
    Builder* algorithm(ParquetCipher::type parquetCipher) {
      parquetCipher_ = parquetCipher;
      return this;
    }

    /// Set a key retrieval metadata (converted from String).
    /// Use either footer_key_metadata or footer_key_id, not both.
    Builder* footerKeyId(const std::string& keyId);

    /// Set a key retrieval metadata.
    /// Use either footer_key_metadata or footer_key_id, not both.
    Builder* footerKeyMetadata(const std::string& footerKeyMetadata);

    /// Set the file AAD Prefix.
    Builder* aadPrefix(const std::string& aadPrefix);

    /// Skip storing AAD Prefix in file.
    /// If not called, and if AAD Prefix is set, it will be stored.
    Builder* disableAadPrefixStorage();

    /// Set the list of encrypted columns and their properties (keys, etc.).
    /// If not called, all columns will be encrypted with the footer key.
    /// If called, the file columns not in the list will be left unencrypted.
    Builder* encryptedColumns(
        const ColumnPathToEncryptionPropertiesMap& encryptedColumns);

    std::shared_ptr<FileEncryptionProperties> build() {
      return std::shared_ptr<FileEncryptionProperties>(
          new FileEncryptionProperties(
              parquetCipher_,
              footerKey_,
              footerKeyMetadata_,
              encryptedFooter_,
              aadPrefix_,
              storeAadPrefixInFile_,
              encryptedColumns_));
    }

   private:
    ParquetCipher::type parquetCipher_;
    bool encryptedFooter_;
    std::string footerKey_;
    std::string footerKeyMetadata_;

    std::string aadPrefix_;
    bool storeAadPrefixInFile_;
    ColumnPathToEncryptionPropertiesMap encryptedColumns_;
  };
  bool encryptedFooter() const {
    return encryptedFooter_;
  }

  EncryptionAlgorithm algorithm() const {
    return algorithm_;
  }

  std::string footerKey() const {
    return footerKey_;
  }

  std::string footerKeyMetadata() const {
    return footerKeyMetadata_;
  }

  std::string fileAad() const {
    return fileAad_;
  }

  std::shared_ptr<ColumnEncryptionProperties> columnEncryptionProperties(
      const std::string& columnPath);

  bool isUtilized() const {
    return utilized_;
  }

  /// FileEncryptionProperties object can be used for writing one file only.
  /// Mark FileEncryptionProperties as utilized once it is used to write a file
  /// as the encryption keys will be wiped out upon completion of file writing.
  void setUtilized() {
    utilized_ = true;
  }

  /// Upon completion of file writing, the encryption keys
  /// will be wiped out (array values set to 0).
  void wipeOutEncryptionKeys();

  /// FileEncryptionProperties object can be used for writing one file only
  /// (at the end, keys are wiped out in the memory).
  /// This method allows cloning identical properties for another file,
  /// with an option to update the aadPrefix (if newAadPrefix is null,
  /// aadPrefix will be cloned too).
  std::shared_ptr<FileEncryptionProperties> deepClone(
      std::string newAadPrefix = "");

  ColumnPathToEncryptionPropertiesMap encryptedColumns() const {
    return encryptedColumns_;
  }

 private:
  EncryptionAlgorithm algorithm_;
  std::string footerKey_;
  std::string footerKeyMetadata_;
  bool encryptedFooter_;
  std::string fileAad_;
  std::string aadPrefix_;
  bool utilized_;
  bool storeAadPrefixInFile_;
  ColumnPathToEncryptionPropertiesMap encryptedColumns_;

  FileEncryptionProperties(
      ParquetCipher::type cipher,
      const std::string& footerKey,
      const std::string& footerKeyMetadata,
      bool encryptedFooter,
      const std::string& aadPrefix,
      bool storeAadPrefixInFile,
      const ColumnPathToEncryptionPropertiesMap& encryptedColumns);
};

} // namespace facebook::velox::parquet::arrow
