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
#include "velox/dwio/parquet/crypto/AesEncryption.h"
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <cstring>
#include <limits>
#include <sstream>
#include "velox/dwio/parquet/crypto/Exception.h"

namespace facebook::velox::parquet {

constexpr int32_t kGcmMode = 0;
constexpr int32_t kCtrMode = 1;
constexpr int32_t kCtrIvLength = 16;

constexpr int32_t AES_128_KEY_LEN = 16;
constexpr int32_t AES_192_KEY_LEN = 24;
constexpr int32_t AES_256_KEY_LEN = 32;

#define ENCRYPT_INIT(CTX, ALG)                                        \
  if (1 != EVP_EncryptInit_ex(CTX, ALG, nullptr, nullptr, nullptr)) { \
    throw CryptoException("Couldn't init ALG encryption");            \
  }

#define DECRYPT_INIT(CTX, ALG)                                        \
  if (1 != EVP_DecryptInit_ex(CTX, ALG, nullptr, nullptr, nullptr)) { \
    throw CryptoException("Couldn't init ALG decryption");            \
  }

class AesDecryptor::AesDecryptorImpl {
 public:
  explicit AesDecryptorImpl(
      ParquetCipher::type algId,
      int32_t keyLen,
      bool metadata,
      bool containsLength);

  ~AesDecryptorImpl() {
    wipeOut();
  }

  int32_t decrypt(
      const uint8_t* ciphertext,
      int32_t ciphertextLen,
      const uint8_t* key,
      int32_t keyLen,
      const uint8_t* aad,
      int32_t aadLen,
      uint8_t* plaintext,
      int32_t plaintextLen);

  void wipeOut() {
    if (nullptr != ctx_) {
      EVP_CIPHER_CTX_free(ctx_);
      ctx_ = nullptr;
    }
  }

  [[nodiscard]] int32_t plaintextLength(int32_t ciphertextLen) const {
    if (ciphertextLen < ciphertextSizeDelta_) {
      std::stringstream ss;
      ss << "Ciphertext length " << ciphertextLen
         << " is invalid, expected at least " << ciphertextSizeDelta_;
      throw CryptoException(ss.str());
    }
    return ciphertextLen - ciphertextSizeDelta_;
  }

  [[nodiscard]] int32_t ciphertextLength(int32_t plaintextLen) const {
    if (plaintextLen < 0) {
      std::stringstream ss;
      ss << "Negative plaintext length " << plaintextLen;
      throw CryptoException(ss.str());
    }
    if (plaintextLen >
        std::numeric_limits<int32_t>::max() - ciphertextSizeDelta_) {
      std::stringstream ss;
      ss << "Plaintext length " << plaintextLen << " plus ciphertext size delta "
         << ciphertextSizeDelta_ << " overflows int32";
      throw CryptoException(ss.str());
    }

    return plaintextLen + ciphertextSizeDelta_;
  }

  /// Get the actual ciphertext length, inclusive of the length buffer length,
  /// and validate that the provided buffer size is large enough.
  [[nodiscard]] int32_t getCiphertextLengthAndValidate(
      const uint8_t* ciphertextBuffer,
      int32_t bufferLen) const;

  /// Get the actual ciphertext length, inclusive of the length buffer length.
  [[nodiscard]] int32_t getCiphertextLength(
      const uint8_t* ciphertextBuffer,
      int32_t bufferLen) const;

 private:
  EVP_CIPHER_CTX* ctx_;
  int32_t aesMode_;
  int32_t keyLength_;
  int32_t ciphertextSizeDelta_;
  int32_t lengthBufferLength_;

  int gcmDecrypt(
      const uint8_t* ciphertext,
      int32_t ciphertextBufferLen,
      const uint8_t* key,
      int32_t keyBufferLen,
      const uint8_t* aad,
      int32_t aadBufferLen,
      uint8_t* plaintext,
      int32_t plaintextBufferLen);

  int32_t ctrDecrypt(
      const uint8_t* ciphertext,
      int32_t ciphertextBufferLen,
      const uint8_t* key,
      int32_t keyBufferLen,
      uint8_t* plaintext,
      int32_t plaintextBufferLen);
};

int32_t AesDecryptor::decrypt(
    const uint8_t* ciphertext,
    int32_t ciphertextBufferLen,
    const uint8_t* key,
    int32_t keyBufferLen,
    const uint8_t* aad,
    int32_t aadBufferLen,
    uint8_t* plaintext,
    int32_t plaintextBufferLen) {
  return impl_->decrypt(
      ciphertext,
      ciphertextBufferLen,
      key,
      keyBufferLen,
      aad,
      aadBufferLen,
      plaintext,
      plaintextBufferLen);
}

void AesDecryptor::wipeOut() {
  impl_->wipeOut();
}

AesDecryptor::~AesDecryptor() {}

AesDecryptor::AesDecryptorImpl::AesDecryptorImpl(
    ParquetCipher::type algId,
    int32_t keyLen,
    bool metadata,
    bool containsLength) {
  ctx_ = nullptr;
  lengthBufferLength_ = containsLength ? kBufferSizeLength : 0;
  ciphertextSizeDelta_ = lengthBufferLength_ + kNonceLength;
  if (metadata || (ParquetCipher::AES_GCM_V1 == algId)) {
    aesMode_ = kGcmMode;
    ciphertextSizeDelta_ += kGcmTagLength;
  } else {
    aesMode_ = kCtrMode;
  }

  if (AES_128_KEY_LEN != keyLen && AES_192_KEY_LEN != keyLen && AES_256_KEY_LEN != keyLen) {
    std::stringstream ss;
    ss << "Wrong key length: " << keyLen;
    throw CryptoException(ss.str());
  }

  keyLength_ = keyLen;

  ctx_ = EVP_CIPHER_CTX_new();
  if (nullptr == ctx_) {
    throw CryptoException("Couldn't init cipher context");
  }

  if (kGcmMode == aesMode_) {
    // Init AES-GCM with specified key length
    if (AES_128_KEY_LEN == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_128_gcm());
    } else if (AES_192_KEY_LEN == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_192_gcm());
    } else if (AES_256_KEY_LEN == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_256_gcm());
    }
  } else {
    // Init AES-CTR with specified key length
    if (AES_128_KEY_LEN == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_128_ctr());
    } else if (AES_192_KEY_LEN == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_192_ctr());
    } else if (AES_256_KEY_LEN == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_256_ctr());
    }
  }
}

AesDecryptor::AesDecryptor(
    ParquetCipher::type algId,
    int32_t keyLen,
    bool metadata,
    bool containsLength)
    : impl_{std::make_unique<AesDecryptorImpl>(
          algId, keyLen, metadata, containsLength)} {}

std::shared_ptr<AesDecryptor>
AesDecryptor::make(ParquetCipher::type algId, int32_t keyLen, bool metadata) {
  if (ParquetCipher::AES_GCM_V1 != algId &&
      ParquetCipher::AES_GCM_CTR_V1 != algId) {
    std::stringstream ss;
    ss << "Crypto algorithm " << algId << " is not supported";
    throw CryptoException(ss.str());
  }

  auto decryptor = std::make_shared<AesDecryptor>(algId, keyLen, metadata);
  return decryptor;
}

int32_t AesDecryptor::plaintextLength(int32_t ciphertextLen) const {
  return impl_->plaintextLength(ciphertextLen);
}

int32_t AesDecryptor::ciphertextLength(int32_t plaintextLen) const {
  return impl_->ciphertextLength(plaintextLen);
}

int32_t AesDecryptor::getCiphertextLengthAndValidate(
    const uint8_t* ciphertext,
    int32_t ciphertextLen) const {
  return impl_->getCiphertextLengthAndValidate(ciphertext, ciphertextLen);
}

int32_t AesDecryptor::getCiphertextLength(
    const uint8_t* ciphertext,
    int32_t ciphertextLen) const {
  return impl_->getCiphertextLength(ciphertext, ciphertextLen);
}

int32_t AesDecryptor::AesDecryptorImpl::getCiphertextLengthAndValidate(
    const uint8_t* ciphertextBuffer,
    int32_t bufferLen) const {
  int32_t ciphertextLength = getCiphertextLength(ciphertextBuffer, bufferLen);
  if (bufferLen < ciphertextLength) {
    std::stringstream ss;
    ss << "Serialized ciphertext length "
       << ciphertextLength
       << " is greater than the provided ciphertext buffer length "
       << bufferLen;
    throw CryptoException(ss.str());
  }
  return ciphertextLength;
}

int32_t AesDecryptor::AesDecryptorImpl::getCiphertextLength(
    const uint8_t* ciphertextBuffer,
    int32_t bufferLen) const {
  if (lengthBufferLength_ > 0) {
    // Note: lengthBufferLength_ must be either 0 or kBufferSizeLength
    if (bufferLen < kBufferSizeLength) {
      std::stringstream ss;
      ss << "Ciphertext buffer length " << bufferLen
         << " is insufficient to read the ciphertext length." << " At least "
         << kBufferSizeLength << " bytes are required.";
      throw CryptoException(ss.str());
    }

    // Extract ciphertext length
    uint32_t writtenCiphertextLen = (static_cast<uint32_t>(ciphertextBuffer[3]) << 24) |
                                    (static_cast<uint32_t>(ciphertextBuffer[2]) << 16) |
                                    (static_cast<uint32_t>(ciphertextBuffer[1]) << 8) |
                                    (static_cast<uint32_t>(ciphertextBuffer[0]));

    if (writtenCiphertextLen >
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max() -
                              lengthBufferLength_)) {
      std::stringstream ss;
      ss << "Written ciphertext length " << writtenCiphertextLen
         << " plus length buffer length " << lengthBufferLength_ << " overflows int32";
      throw CryptoException(ss.str());
    }
    if (bufferLen <
        static_cast<size_t>(writtenCiphertextLen) + lengthBufferLength_) {
      std::stringstream ss;
      ss << "Serialized ciphertext length "
         << (writtenCiphertextLen + lengthBufferLength_)
         << " is greater than the provided ciphertext buffer length "
         << bufferLen;
      throw CryptoException(ss.str());
    }

    return static_cast<int32_t>(writtenCiphertextLen) + lengthBufferLength_;
  }
  if (bufferLen > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    std::stringstream ss;
    ss << "Ciphertext buffer length " << bufferLen << " overflows int32";
    throw CryptoException(ss.str());
  }
  return bufferLen;
}

int32_t AesDecryptor::AesDecryptorImpl::gcmDecrypt(
    const uint8_t* ciphertext,
    int32_t ciphertextBufferLen,
    const uint8_t* key,
    int32_t keyBufferLen,
    const uint8_t* aad,
    int32_t aadBufferLen,
    uint8_t* plaintext,
    int32_t plaintextBufferLen) {
  int len;
  int32_t plaintextLen;

  uint8_t tag[kGcmTagLength];
  memset(tag, 0, kGcmTagLength);
  uint8_t nonce[kNonceLength];
  memset(nonce, 0, kNonceLength);

  int32_t ciphertextLen = getCiphertextLengthAndValidate(ciphertext, ciphertextBufferLen);

  if (plaintextBufferLen < ciphertextLen - ciphertextSizeDelta_) {
    std::stringstream ss;
    ss << "Plaintext buffer length " << plaintextBufferLen << " is insufficient "
       << "for ciphertext length " << ciphertextLen;
    throw CryptoException(ss.str());
  }

  if (ciphertextLen < lengthBufferLength_ + kNonceLength + kGcmTagLength) {
    std::stringstream ss;
    ss << "Invalid ciphertext length " << ciphertextLen
       << ". Expected at least "
       << lengthBufferLength_ + kNonceLength + kGcmTagLength << "\n";
    throw CryptoException(ss.str());
  }

  // Extracting IV and tag
  std::copy(
      ciphertext + lengthBufferLength_,
      ciphertext + lengthBufferLength_ + kNonceLength,
      nonce);
  std::copy(
      ciphertext + ciphertextLen - kGcmTagLength,
      ciphertext + ciphertextLen,
      tag);

  // Setting key and IV
  if (1 != EVP_DecryptInit_ex(ctx_, nullptr, nullptr, key, nonce)) {
    throw CryptoException("Couldn't set key and IV");
  }

  // Setting additional authenticated data
  if (aadBufferLen > static_cast<size_t>(std::numeric_limits<int>::max())) {
    std::stringstream ss;
    ss << "AAD size " << aadBufferLen << " overflows int";
    throw CryptoException(ss.str());
  }

  // Setting additional authenticated data
  if (aad && aadBufferLen > 0 &&
      (1 != EVP_DecryptUpdate(ctx_, nullptr, &len, aad, aadBufferLen))) {
    throw CryptoException("Couldn't set AAD");
  }

  // Decryption
  if (!EVP_DecryptUpdate(
          ctx_,
          plaintext,
          &len,
          ciphertext + lengthBufferLength_ + kNonceLength,
          ciphertextLen - lengthBufferLength_ - kNonceLength -
              kGcmTagLength)) {
    throw CryptoException("Failed decryption update gcm");
  }

  plaintextLen = len;

  // Checking the tag (authentication)
  if (!EVP_CIPHER_CTX_ctrl(ctx_, EVP_CTRL_GCM_SET_TAG, kGcmTagLength, tag)) {
    throw CryptoException("Failed authentication");
  }

  // Finalization
  if (1 != EVP_DecryptFinal_ex(ctx_, plaintext + len, &len)) {
    throw CryptoException("Failed decryption finalization gcm");
  }

  plaintextLen += len;
  return plaintextLen;
}

int32_t AesDecryptor::AesDecryptorImpl::ctrDecrypt(
    const uint8_t* ciphertext,
    int32_t ciphertextBufferLen,
    const uint8_t* key,
    int32_t keyBufferLen,
    uint8_t* plaintext,
    int32_t plaintextBufferLen) {
  int len;
  int32_t plaintextLen;

  uint8_t iv[kCtrIvLength];
  memset(iv, 0, kCtrIvLength);

  int32_t ciphertextLen = getCiphertextLengthAndValidate(ciphertext, ciphertextBufferLen);

  if (plaintextBufferLen < ciphertextBufferLen - ciphertextSizeDelta_) {
    std::stringstream ss;
    ss << "Plaintext buffer length " << plaintextBufferLen << " is insufficient "
       << "for ciphertext length " << ciphertextLen;
    throw CryptoException(ss.str());
  }

  if (ciphertextLen < lengthBufferLength_ + kNonceLength) {
    std::stringstream ss;
    ss << "Invalid ciphertext length " << ciphertextLen
       << ". Expected at least " << lengthBufferLength_ + kNonceLength << "\n";
    throw CryptoException(ss.str());
  }

  // Extracting nonce
  std::copy(
      ciphertext + lengthBufferLength_,
      ciphertext + lengthBufferLength_ + kNonceLength,
      iv);
  // Parquet CTR IVs are comprised of a 12-byte nonce and a 4-byte initial
  // counter field.
  // The first 31 bits of the initial counter field are set to 0, the last bit
  // is set to 1.
  iv[kCtrIvLength - 1] = 1;

  // Setting key and IV
  if (1 != EVP_DecryptInit_ex(ctx_, nullptr, nullptr, key, iv)) {
    throw CryptoException("Couldn't set key and IV");
  }

  // Decryption
  if (!EVP_DecryptUpdate(
          ctx_,
          plaintext,
          &len,
          ciphertext + lengthBufferLength_ + kNonceLength,
          ciphertextLen - lengthBufferLength_ - kNonceLength)) {
    throw CryptoException("Failed decryption update ctr");
  }

  plaintextLen = len;

  // Finalization
  if (1 != EVP_DecryptFinal_ex(ctx_, plaintext + len, &len)) {
    throw CryptoException("Failed decryption finalization ctr");
  }

  plaintextLen += len;
  return plaintextLen;
}

int32_t AesDecryptor::AesDecryptorImpl::decrypt(
    const uint8_t* ciphertext,
    int32_t ciphertextBufferLen,
    const uint8_t* key,
    int32_t keyBufferLen,
    const uint8_t* aad,
    int32_t aadBufferLen,
    uint8_t* plaintext,
    int32_t plaintextBufferLen) {
  if (static_cast<size_t>(keyLength_) != keyBufferLen) {
    std::stringstream ss;
    ss << "Wrong key length " << keyBufferLen << ". Should be " << keyLength_;
    throw CryptoException(ss.str());
  }

  if (kGcmMode == aesMode_) {
    return gcmDecrypt(
        ciphertext,
        ciphertextBufferLen,
        key,
        keyBufferLen,
        aad,
        aadBufferLen,
        plaintext,
        plaintextBufferLen);
  }

  return ctrDecrypt(
      ciphertext, ciphertextBufferLen, key, keyBufferLen, plaintext, plaintextBufferLen);
}

static std::string shortToBytesLe(int16_t input) {
  int8_t output[2];
  memset(output, 0, 2);
  output[1] = static_cast<int8_t>(0xff & (input >> 8));
  output[0] = static_cast<int8_t>(0xff & (input));

  return std::string(reinterpret_cast<char const*>(output), 2);
}

static void CheckPageOrdinal(int32_t pageOrdinal) {
  if (pageOrdinal > std::numeric_limits<int16_t>::max()) {
    throw CryptoException(
        "Encrypted Parquet files can't have more than " +
        std::to_string(std::numeric_limits<int16_t>::max()) +
        " pages per chunk: got " + std::to_string(pageOrdinal));
  }
}

std::string createModuleAad(
    const std::string& fileAad,
    int8_t moduleType,
    int16_t rowGroupOrdinal,
    int16_t columnOrdinal,
    int16_t pageOrdinal) {
  CheckPageOrdinal(pageOrdinal);
  int8_t typeOrdinalBytes[1];
  typeOrdinalBytes[0] = moduleType;
  std::string typeOrdinalBytesStr(
      reinterpret_cast<char const*>(typeOrdinalBytes), 1);
  if (kFooter == moduleType) {
    std::string result = fileAad + typeOrdinalBytesStr;
    return result;
  }
  std::string rowGroupOrdinalBytes = shortToBytesLe(rowGroupOrdinal);
  std::string columnOrdinalBytes = shortToBytesLe(columnOrdinal);
  if (kDataPage != moduleType && kDataPageHeader != moduleType) {
    std::ostringstream out;
    out << fileAad << typeOrdinalBytesStr << rowGroupOrdinalBytes
        << columnOrdinalBytes;
    return out.str();
  }
  std::string pageOrdinalBytes = shortToBytesLe(pageOrdinal);
  std::ostringstream out;
  out << fileAad << typeOrdinalBytesStr << rowGroupOrdinalBytes
      << columnOrdinalBytes << pageOrdinalBytes;
  return out.str();
}

} // namespace facebook::velox::parquet
