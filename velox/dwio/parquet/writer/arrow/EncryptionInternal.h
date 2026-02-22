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

#include <memory>
#include <string>
#include <vector>

#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

using facebook::velox::parquet::arrow::ParquetCipher;

namespace facebook::velox::parquet::arrow::encryption {

constexpr int kGcmTagLength = 16;
constexpr int kNonceLength = 12;

// Module types.
constexpr int8_t kFooter = 0;
constexpr int8_t kColumnMetaData = 1;
constexpr int8_t kDataPage = 2;
constexpr int8_t kDictionaryPage = 3;
constexpr int8_t kDataPageHeader = 4;
constexpr int8_t kDictionaryPageHeader = 5;
constexpr int8_t kColumnIndex = 6;
constexpr int8_t kOffsetIndex = 7;

/// Performs AES encryption operations with GCM or CTR ciphers.
class AesEncryptor {
 public:
  /// Can serve one key length only. Possible values: 16, 24, 32 bytes.
  /// If write_length is true, prepend ciphertext length to the ciphertext.
  explicit AesEncryptor(
      ParquetCipher::type algId,
      int keyLen,
      bool metadata,
      bool writeLength = true);

  static AesEncryptor* make(
      ParquetCipher::type algId,
      int keyLen,
      bool metadata,
      std::vector<AesEncryptor*>* allEncryptors);

  static AesEncryptor* make(
      ParquetCipher::type algId,
      int keyLen,
      bool metadata,
      bool writeLength,
      std::vector<AesEncryptor*>* allEncryptors);

  ~AesEncryptor();

  /// Size difference between plaintext and ciphertext, for this cipher.
  int ciphertextSizeDelta();

  /// Encrypts plaintext with the key and aad. Key length is passed only for.
  /// Validation. If different from value in constructor, exception will be.
  /// Thrown.
  int encrypt(
      const uint8_t* plaintext,
      int plaintextLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* aad,
      int aadLen,
      uint8_t* ciphertext);

  /// Encrypts plaintext footer, in order to compute footer signature (tag).
  int signedFooterEncrypt(
      const uint8_t* footer,
      int footerLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* aad,
      int aadLen,
      const uint8_t* nonce,
      uint8_t* encryptedFooter);

  void wipeOut();

 private:
  // PIMPL Idiom.
  class AesEncryptorImpl;
  std::unique_ptr<AesEncryptorImpl> impl_;
};

/// Performs AES decryption operations with GCM or CTR ciphers.
class AesDecryptor {
 public:
  /// Can serve one key length only. Possible values: 16, 24, 32 bytes.
  /// If contains_length is true, expect ciphertext length prepended to the.
  /// Ciphertext.
  explicit AesDecryptor(
      ParquetCipher::type algId,
      int keyLen,
      bool metadata,
      bool containsLength = true);

  /// \brief Factory function to create an AesDecryptor.
  ///
  /// \param alg_id the encryption algorithm to use.
  /// \param key_len key length. Possible values: 16, 24, 32 bytes.
  /// \param metadata if true then this is a metadata decryptor.
  /// \param all_decryptors A weak reference to all decryptors that need to be.
  /// Wiped out when decryption is finished \return shared pointer to a new.
  /// AesDecryptor.
  static std::shared_ptr<AesDecryptor> make(
      ParquetCipher::type algId,
      int keyLen,
      bool metadata,
      std::vector<std::weak_ptr<AesDecryptor>>* allDecryptors);

  ~AesDecryptor();
  void wipeOut();

  /// Size difference between plaintext and ciphertext, for this cipher.
  int ciphertextSizeDelta();

  /// Decrypts ciphertext with the key and aad. Key length is passed only for.
  /// Validation. If different from value in constructor, exception will be.
  /// Thrown.
  int decrypt(
      const uint8_t* ciphertext,
      int ciphertextLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* aad,
      int aadLen,
      uint8_t* plaintext);

 private:
  // PIMPL Idiom.
  class AesDecryptorImpl;
  std::unique_ptr<AesDecryptorImpl> impl_;
};

std::string createModuleAad(
    const std::string& fileAad,
    int8_t moduleType,
    int16_t rowGroupOrdinal,
    int16_t columnOrdinal,
    int32_t pageOrdinal);

std::string createFooterAad(const std::string& aadPrefixBytes);

// Update last two bytes of page (or page header) module AAD.
void quickUpdatePageAad(int32_t newPageOrdinal, std::string* AAD);

// Wraps OpenSSL RAND_bytes function.
void randBytes(unsigned char* buf, int num);

} // namespace facebook::velox::parquet::arrow::encryption
