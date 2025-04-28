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
#include <memory>
#include <vector>

namespace facebook::velox::parquet {

struct ParquetCipher {
  enum type { AES_GCM_V1 = 0, AES_GCM_CTR_V1 = 1 };
};

// encryption size length
constexpr int kBufferSizeLength = 4;

constexpr int kGcmTagLength = 16;
constexpr int kNonceLength = 12;

// Module types
constexpr int8_t kFooter = 0;
constexpr int8_t kColumnMetaData = 1;
constexpr int8_t kDataPage = 2;
constexpr int8_t kDictionaryPage = 3;
constexpr int8_t kDataPageHeader = 4;
constexpr int8_t kDictionaryPageHeader = 5;
constexpr int8_t kColumnIndex = 6;
constexpr int8_t kOffsetIndex = 7;
constexpr int8_t kBloomFilterHeader = 8;
constexpr int8_t kBloomFilterBitset = 9;

// Performs AES decryption operations with GCM or CTR ciphers.
// It refers to the Apache Arrow's implementation at https://github.com/apache/arrow/blob/main/cpp/src/parquet/encryption/encryption_internal.h
class AesDecryptor {
 public:
  explicit AesDecryptor(ParquetCipher::type algId, int keyLen, bool metadata,
                        bool containsLength = true);

  static std::shared_ptr<AesDecryptor> make(
      ParquetCipher::type algId, int keyLen, bool metadata);

  ~AesDecryptor();
  void wipeOut();

  [[nodiscard]] int plaintextLength(int ciphertextLen) const;

  [[nodiscard]] int ciphertextLength(int plaintextLen) const;

  int getCiphertextLength(const uint8_t* ciphertext, int ciphertextLen) const;

  int getCiphertextLengthWithoutValidation(const uint8_t* ciphertext, int ciphertextLen) const;

  /// Decrypts ciphertext with the key and aad. Key length is passed only for
  /// validation. If different from value in constructor, exception will be thrown.
  /// The caller is responsible for ensuring that the plaintext buffer is at least as
  /// large as PlaintextLength(ciphertext_len).
  int decrypt(const uint8_t* ciphertext, int ciphertextLen,
              const uint8_t* key, int keyLen,
              const uint8_t* aad, int aadLen,
              uint8_t* plaintext, int plaintextLen);

 private:
  class AesDecryptorImpl;
  std::unique_ptr<AesDecryptorImpl> impl_;
};

std::string createModuleAad(const std::string& fileAad, int8_t moduleType,
                            int16_t rowGroupOrdinal, int16_t columnOrdinal,
                            int16_t pageOrdinal);

}
