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

#include "velox/dwio/parquet/writer/arrow/EncryptionInternal.h"
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "velox/dwio/parquet/writer/arrow/Exception.h"

namespace facebook::velox::parquet::arrow::encryption {

constexpr int kGcmMode = 0;
constexpr int kCtrMode = 1;
constexpr int kCtrIvLength = 16;
constexpr int kBufferSizeLength = 4;

#define ENCRYPT_INIT(CTX, ALG)                                          \
  do {                                                                  \
    if (1 != EVP_EncryptInit_ex(CTX, ALG, nullptr, nullptr, nullptr)) { \
      throw ParquetException("Couldn't init ALG encryption");           \
    }                                                                   \
  } while (0)

#define DECRYPT_INIT(CTX, ALG)                                          \
  do {                                                                  \
    if (1 != EVP_DecryptInit_ex(CTX, ALG, nullptr, nullptr, nullptr)) { \
      throw ParquetException("Couldn't init ALG decryption");           \
    }                                                                   \
  } while (0)

class AesEncryptor::AesEncryptorImpl {
 public:
  explicit AesEncryptorImpl(
      ParquetCipher::type algId,
      int keyLen,
      bool metadata,
      bool writeLength);

  ~AesEncryptorImpl() {
    if (nullptr != ctx_) {
      EVP_CIPHER_CTX_free(ctx_);
      ctx_ = nullptr;
    }
  }

  int encrypt(
      const uint8_t* plaintext,
      int plaintextLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* aad,
      int aadLen,
      uint8_t* ciphertext);

  int signedFooterEncrypt(
      const uint8_t* footer,
      int footerLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* aad,
      int aadLen,
      const uint8_t* nonce,
      uint8_t* encryptedFooter);
  void wipeOut() {
    if (nullptr != ctx_) {
      EVP_CIPHER_CTX_free(ctx_);
      ctx_ = nullptr;
    }
  }

  int ciphertextSizeDelta() {
    return ciphertextSizeDelta_;
  }

 private:
  EVP_CIPHER_CTX* ctx_;
  int aesMode_;
  int keyLength_;
  int ciphertextSizeDelta_;
  int lengthBufferLength_;

  int gcmEncrypt(
      const uint8_t* plaintext,
      int plaintextLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* nonce,
      const uint8_t* aad,
      int aadLen,
      uint8_t* ciphertext);

  int ctrEncrypt(
      const uint8_t* plaintext,
      int plaintextLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* nonce,
      uint8_t* ciphertext);
};

AesEncryptor::AesEncryptorImpl::AesEncryptorImpl(
    ParquetCipher::type algId,
    int keyLen,
    bool metadata,
    bool writeLength) {
  ctx_ = nullptr;

  lengthBufferLength_ = writeLength ? kBufferSizeLength : 0;
  ciphertextSizeDelta_ = lengthBufferLength_ + kNonceLength;
  if (metadata || (ParquetCipher::kAesGcmV1 == algId)) {
    aesMode_ = kGcmMode;
    ciphertextSizeDelta_ += kGcmTagLength;
  } else {
    aesMode_ = kCtrMode;
  }

  if (16 != keyLen && 24 != keyLen && 32 != keyLen) {
    std::stringstream ss;
    ss << "Wrong key length: " << keyLen;
    throw ParquetException(ss.str());
  }

  keyLength_ = keyLen;

  ctx_ = EVP_CIPHER_CTX_new();
  if (nullptr == ctx_) {
    throw ParquetException("Couldn't init cipher context");
  }

  if (kGcmMode == aesMode_) {
    // Init AES-GCM with specified key length.
    if (16 == keyLen) {
      ENCRYPT_INIT(ctx_, EVP_aes_128_gcm());
    } else if (24 == keyLen) {
      ENCRYPT_INIT(ctx_, EVP_aes_192_gcm());
    } else if (32 == keyLen) {
      ENCRYPT_INIT(ctx_, EVP_aes_256_gcm());
    }
  } else {
    // Init AES-CTR with specified key length.
    if (16 == keyLen) {
      ENCRYPT_INIT(ctx_, EVP_aes_128_ctr());
    } else if (24 == keyLen) {
      ENCRYPT_INIT(ctx_, EVP_aes_192_ctr());
    } else if (32 == keyLen) {
      ENCRYPT_INIT(ctx_, EVP_aes_256_ctr());
    }
  }
}

int AesEncryptor::AesEncryptorImpl::signedFooterEncrypt(
    const uint8_t* footer,
    int footerLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    const uint8_t* nonce,
    uint8_t* encryptedFooter) {
  if (keyLength_ != keyLen) {
    std::stringstream ss;
    ss << "Wrong key length " << keyLen << ". Should be " << keyLength_;
    throw ParquetException(ss.str());
  }

  if (kGcmMode != aesMode_) {
    throw ParquetException("Must use AES GCM (metadata) encryptor");
  }

  return gcmEncrypt(
      footer, footerLen, key, keyLen, nonce, aad, aadLen, encryptedFooter);
}

int AesEncryptor::AesEncryptorImpl::encrypt(
    const uint8_t* plaintext,
    int plaintextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    uint8_t* ciphertext) {
  if (keyLength_ != keyLen) {
    std::stringstream ss;
    ss << "Wrong key length " << keyLen << ". Should be " << keyLength_;
    throw ParquetException(ss.str());
  }

  uint8_t nonce[kNonceLength];
  memset(nonce, 0, kNonceLength);
  // Random nonce.
  randBytes(nonce, sizeof(nonce));

  if (kGcmMode == aesMode_) {
    return gcmEncrypt(
        plaintext, plaintextLen, key, keyLen, nonce, aad, aadLen, ciphertext);
  }

  return ctrEncrypt(plaintext, plaintextLen, key, keyLen, nonce, ciphertext);
}

int AesEncryptor::AesEncryptorImpl::gcmEncrypt(
    const uint8_t* plaintext,
    int plaintextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* nonce,
    const uint8_t* aad,
    int aadLen,
    uint8_t* ciphertext) {
  int len = 0;
  int ciphertextLen;

  uint8_t tag[kGcmTagLength];
  memset(tag, 0, kGcmTagLength);

  // Setting key and IV (nonce)
  if (1 != EVP_EncryptInit_ex(ctx_, nullptr, nullptr, key, nonce)) {
    throw ParquetException("Couldn't set key and nonce");
  }

  // Setting additional authenticated data.
  if ((nullptr != aad) &&
      (1 != EVP_EncryptUpdate(ctx_, nullptr, &len, aad, aadLen))) {
    throw ParquetException("Couldn't set AAD");
  }

  // Encryption.
  if (1 !=
      EVP_EncryptUpdate(
          ctx_,
          ciphertext + lengthBufferLength_ + kNonceLength,
          &len,
          plaintext,
          plaintextLen)) {
    throw ParquetException("Failed encryption update");
  }

  ciphertextLen = len;

  // Finalization.
  if (1 !=
      EVP_EncryptFinal_ex(
          ctx_, ciphertext + lengthBufferLength_ + kNonceLength + len, &len)) {
    throw ParquetException("Failed encryption finalization");
  }

  ciphertextLen += len;

  // Getting the tag.
  if (1 !=
      EVP_CIPHER_CTX_ctrl(ctx_, EVP_CTRL_GCM_GET_TAG, kGcmTagLength, tag)) {
    throw ParquetException("Couldn't get AES-GCM tag");
  }

  // Copying the buffer size, nonce and tag to ciphertext.
  uint32_t bufferSize = kNonceLength + ciphertextLen + kGcmTagLength;
  if (lengthBufferLength_ > 0) {
    ciphertext[3] = static_cast<uint8_t>(0xff & (bufferSize >> 24));
    ciphertext[2] = static_cast<uint8_t>(0xff & (bufferSize >> 16));
    ciphertext[1] = static_cast<uint8_t>(0xff & (bufferSize >> 8));
    ciphertext[0] = static_cast<uint8_t>(0xff & (bufferSize));
  }
  std::copy(nonce, nonce + kNonceLength, ciphertext + lengthBufferLength_);
  std::copy(
      tag,
      tag + kGcmTagLength,
      ciphertext + lengthBufferLength_ + kNonceLength + ciphertextLen);

  return lengthBufferLength_ + bufferSize;
}

int AesEncryptor::AesEncryptorImpl::ctrEncrypt(
    const uint8_t* plaintext,
    int plaintextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* nonce,
    uint8_t* ciphertext) {
  int len = 0;
  int ciphertextLen;

  // Parquet CTR IVs are comprised of a 12-byte nonce and a 4-byte initial.
  // Counter field.
  // The first 31 bits of the initial counter field are set to 0, the last bit.
  // Is set to 1.
  uint8_t iv[kCtrIvLength];
  memset(iv, 0, kCtrIvLength);
  std::copy(nonce, nonce + kNonceLength, iv);
  iv[kCtrIvLength - 1] = 1;

  // Setting key and IV.
  if (1 != EVP_EncryptInit_ex(ctx_, nullptr, nullptr, key, iv)) {
    throw ParquetException("Couldn't set key and IV");
  }

  // Encryption.
  if (1 !=
      EVP_EncryptUpdate(
          ctx_,
          ciphertext + lengthBufferLength_ + kNonceLength,
          &len,
          plaintext,
          plaintextLen)) {
    throw ParquetException("Failed encryption update");
  }

  ciphertextLen = len;

  // Finalization.
  if (1 !=
      EVP_EncryptFinal_ex(
          ctx_, ciphertext + lengthBufferLength_ + kNonceLength + len, &len)) {
    throw ParquetException("Failed encryption finalization");
  }

  ciphertextLen += len;

  // Copying the buffer size and nonce to ciphertext.
  uint32_t bufferSize = kNonceLength + ciphertextLen;
  if (lengthBufferLength_ > 0) {
    ciphertext[3] = static_cast<uint8_t>(0xff & (bufferSize >> 24));
    ciphertext[2] = static_cast<uint8_t>(0xff & (bufferSize >> 16));
    ciphertext[1] = static_cast<uint8_t>(0xff & (bufferSize >> 8));
    ciphertext[0] = static_cast<uint8_t>(0xff & (bufferSize));
  }
  std::copy(nonce, nonce + kNonceLength, ciphertext + lengthBufferLength_);

  return lengthBufferLength_ + bufferSize;
}

AesEncryptor::~AesEncryptor() {}

int AesEncryptor::signedFooterEncrypt(
    const uint8_t* footer,
    int footerLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    const uint8_t* nonce,
    uint8_t* encryptedFooter) {
  return impl_->signedFooterEncrypt(
      footer, footerLen, key, keyLen, aad, aadLen, nonce, encryptedFooter);
}

void AesEncryptor::wipeOut() {
  impl_->wipeOut();
}

int AesEncryptor::ciphertextSizeDelta() {
  return impl_->ciphertextSizeDelta();
}

int AesEncryptor::encrypt(
    const uint8_t* plaintext,
    int plaintextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    uint8_t* ciphertext) {
  return impl_->encrypt(
      plaintext, plaintextLen, key, keyLen, aad, aadLen, ciphertext);
}

AesEncryptor::AesEncryptor(
    ParquetCipher::type algId,
    int keyLen,
    bool metadata,
    bool writeLength)
    : impl_{std::unique_ptr<AesEncryptorImpl>(
          new AesEncryptorImpl(algId, keyLen, metadata, writeLength))} {}

class AesDecryptor::AesDecryptorImpl {
 public:
  explicit AesDecryptorImpl(
      ParquetCipher::type algId,
      int keyLen,
      bool metadata,
      bool containsLength);

  ~AesDecryptorImpl() {
    if (nullptr != ctx_) {
      EVP_CIPHER_CTX_free(ctx_);
      ctx_ = nullptr;
    }
  }

  int decrypt(
      const uint8_t* ciphertext,
      int ciphertextLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* aad,
      int aadLen,
      uint8_t* plaintext);

  void wipeOut() {
    if (nullptr != ctx_) {
      EVP_CIPHER_CTX_free(ctx_);
      ctx_ = nullptr;
    }
  }

  int ciphertextSizeDelta() {
    return ciphertextSizeDelta_;
  }

 private:
  EVP_CIPHER_CTX* ctx_;
  int aesMode_;
  int keyLength_;
  int ciphertextSizeDelta_;
  int lengthBufferLength_;
  int gcmDecrypt(
      const uint8_t* ciphertext,
      int ciphertextLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* aad,
      int aadLen,
      uint8_t* plaintext);

  int ctrDecrypt(
      const uint8_t* ciphertext,
      int ciphertextLen,
      const uint8_t* key,
      int keyLen,
      uint8_t* plaintext);
};

int AesDecryptor::decrypt(
    const uint8_t* plaintext,
    int plaintextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    uint8_t* ciphertext) {
  return impl_->decrypt(
      plaintext, plaintextLen, key, keyLen, aad, aadLen, ciphertext);
}

void AesDecryptor::wipeOut() {
  impl_->wipeOut();
}

AesDecryptor::~AesDecryptor() {}

AesDecryptor::AesDecryptorImpl::AesDecryptorImpl(
    ParquetCipher::type algId,
    int keyLen,
    bool metadata,
    bool containsLength) {
  ctx_ = nullptr;
  lengthBufferLength_ = containsLength ? kBufferSizeLength : 0;
  ciphertextSizeDelta_ = lengthBufferLength_ + kNonceLength;
  if (metadata || (ParquetCipher::kAesGcmV1 == algId)) {
    aesMode_ = kGcmMode;
    ciphertextSizeDelta_ += kGcmTagLength;
  } else {
    aesMode_ = kCtrMode;
  }

  if (16 != keyLen && 24 != keyLen && 32 != keyLen) {
    std::stringstream ss;
    ss << "Wrong key length: " << keyLen;
    throw ParquetException(ss.str());
  }

  keyLength_ = keyLen;

  ctx_ = EVP_CIPHER_CTX_new();
  if (nullptr == ctx_) {
    throw ParquetException("Couldn't init cipher context");
  }

  if (kGcmMode == aesMode_) {
    // Init AES-GCM with specified key length.
    if (16 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_128_gcm());
    } else if (24 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_192_gcm());
    } else if (32 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_256_gcm());
    }
  } else {
    // Init AES-CTR with specified key length.
    if (16 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_128_ctr());
    } else if (24 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_192_ctr());
    } else if (32 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_256_ctr());
    }
  }
}

AesEncryptor* AesEncryptor::make(
    ParquetCipher::type algId,
    int keyLen,
    bool metadata,
    std::vector<AesEncryptor*>* allEncryptors) {
  return make(algId, keyLen, metadata, true /*write_length*/, allEncryptors);
}

AesEncryptor* AesEncryptor::make(
    ParquetCipher::type algId,
    int keyLen,
    bool metadata,
    bool writeLength,
    std::vector<AesEncryptor*>* allEncryptors) {
  if (ParquetCipher::kAesGcmV1 != algId &&
      ParquetCipher::kAesGcmCtrV1 != algId) {
    std::stringstream ss;
    ss << "Crypto algorithm " << algId << " is not supported";
    throw ParquetException(ss.str());
  }

  AesEncryptor* Encryptor =
      new AesEncryptor(algId, keyLen, metadata, writeLength);
  if (allEncryptors != nullptr)
    allEncryptors->push_back(Encryptor);
  return Encryptor;
}

AesDecryptor::AesDecryptor(
    ParquetCipher::type algId,
    int keyLen,
    bool metadata,
    bool containsLength)
    : impl_{std::unique_ptr<AesDecryptorImpl>(
          new AesDecryptorImpl(algId, keyLen, metadata, containsLength))} {}

std::shared_ptr<AesDecryptor> AesDecryptor::make(
    ParquetCipher::type algId,
    int keyLen,
    bool metadata,
    std::vector<std::weak_ptr<AesDecryptor>>* allDecryptors) {
  if (ParquetCipher::kAesGcmV1 != algId &&
      ParquetCipher::kAesGcmCtrV1 != algId) {
    std::stringstream ss;
    ss << "Crypto algorithm " << algId << " is not supported";
    throw ParquetException(ss.str());
  }

  auto Decryptor = std::make_shared<AesDecryptor>(algId, keyLen, metadata);
  if (allDecryptors != nullptr) {
    allDecryptors->push_back(Decryptor);
  }
  return Decryptor;
}

int AesDecryptor::ciphertextSizeDelta() {
  return impl_->ciphertextSizeDelta();
}

int AesDecryptor::AesDecryptorImpl::gcmDecrypt(
    const uint8_t* ciphertext,
    int ciphertextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    uint8_t* plaintext) {
  int len = 0;
  int plaintextLen;

  uint8_t tag[kGcmTagLength];
  memset(tag, 0, kGcmTagLength);
  uint8_t nonce[kNonceLength];
  memset(nonce, 0, kNonceLength);

  if (lengthBufferLength_ > 0) {
    // Extract ciphertext length.
    uint32_t writtenCiphertextLen = ((ciphertext[3] & 0xff) << 24) |
        ((ciphertext[2] & 0xff) << 16) | ((ciphertext[1] & 0xff) << 8) |
        ((ciphertext[0] & 0xff));

    if (ciphertextLen > 0 &&
        ciphertextLen != (writtenCiphertextLen + lengthBufferLength_)) {
      throw ParquetException("Wrong ciphertext length");
    }
    ciphertextLen = writtenCiphertextLen + lengthBufferLength_;
  } else {
    if (ciphertextLen == 0) {
      throw ParquetException("Zero ciphertext length");
    }
  }

  // Extracting IV and tag.
  std::copy(
      ciphertext + lengthBufferLength_,
      ciphertext + lengthBufferLength_ + kNonceLength,
      nonce);
  std::copy(
      ciphertext + ciphertextLen - kGcmTagLength,
      ciphertext + ciphertextLen,
      tag);

  // Setting key and IV.
  if (1 != EVP_DecryptInit_ex(ctx_, nullptr, nullptr, key, nonce)) {
    throw ParquetException("Couldn't set key and IV");
  }

  // Setting additional authenticated data.
  if ((nullptr != aad) &&
      (1 != EVP_DecryptUpdate(ctx_, nullptr, &len, aad, aadLen))) {
    throw ParquetException("Couldn't set AAD");
  }

  // Decryption.
  if (!EVP_DecryptUpdate(
          ctx_,
          plaintext,
          &len,
          ciphertext + lengthBufferLength_ + kNonceLength,
          ciphertextLen - lengthBufferLength_ - kNonceLength - kGcmTagLength)) {
    throw ParquetException("Failed decryption update");
  }

  plaintextLen = len;

  // Checking the tag (authentication)
  if (!EVP_CIPHER_CTX_ctrl(ctx_, EVP_CTRL_GCM_SET_TAG, kGcmTagLength, tag)) {
    throw ParquetException("Failed authentication");
  }

  // Finalization.
  if (1 != EVP_DecryptFinal_ex(ctx_, plaintext + len, &len)) {
    throw ParquetException("Failed decryption finalization");
  }

  plaintextLen += len;
  return plaintextLen;
}

int AesDecryptor::AesDecryptorImpl::ctrDecrypt(
    const uint8_t* ciphertext,
    int ciphertextLen,
    const uint8_t* key,
    int keyLen,
    uint8_t* plaintext) {
  int len = 0;
  int plaintextLen;

  uint8_t iv[kCtrIvLength];
  memset(iv, 0, kCtrIvLength);

  if (lengthBufferLength_ > 0) {
    // Extract ciphertext length.
    uint32_t writtenCiphertextLen = ((ciphertext[3] & 0xff) << 24) |
        ((ciphertext[2] & 0xff) << 16) | ((ciphertext[1] & 0xff) << 8) |
        ((ciphertext[0] & 0xff));

    if (ciphertextLen > 0 &&
        ciphertextLen != (writtenCiphertextLen + lengthBufferLength_)) {
      throw ParquetException("Wrong ciphertext length");
    }
    ciphertextLen = writtenCiphertextLen;
  } else {
    if (ciphertextLen == 0) {
      throw ParquetException("Zero ciphertext length");
    }
  }

  // Extracting nonce.
  std::copy(
      ciphertext + lengthBufferLength_,
      ciphertext + lengthBufferLength_ + kNonceLength,
      iv);
  // Parquet CTR IVs are comprised of a 12-byte nonce and a 4-byte initial.
  // Counter field.
  // The first 31 bits of the initial counter field are set to 0, the last bit.
  // Is set to 1.
  iv[kCtrIvLength - 1] = 1;

  // Setting key and IV.
  if (1 != EVP_DecryptInit_ex(ctx_, nullptr, nullptr, key, iv)) {
    throw ParquetException("Couldn't set key and IV");
  }

  // Decryption.
  if (!EVP_DecryptUpdate(
          ctx_,
          plaintext,
          &len,
          ciphertext + lengthBufferLength_ + kNonceLength,
          ciphertextLen - kNonceLength)) {
    throw ParquetException("Failed decryption update");
  }

  plaintextLen = len;

  // Finalization.
  if (1 != EVP_DecryptFinal_ex(ctx_, plaintext + len, &len)) {
    throw ParquetException("Failed decryption finalization");
  }

  plaintextLen += len;
  return plaintextLen;
}

int AesDecryptor::AesDecryptorImpl::decrypt(
    const uint8_t* ciphertext,
    int ciphertextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    uint8_t* plaintext) {
  if (keyLength_ != keyLen) {
    std::stringstream ss;
    ss << "Wrong key length " << keyLen << ". Should be " << keyLength_;
    throw ParquetException(ss.str());
  }

  if (kGcmMode == aesMode_) {
    return gcmDecrypt(
        ciphertext, ciphertextLen, key, keyLen, aad, aadLen, plaintext);
  }

  return ctrDecrypt(ciphertext, ciphertextLen, key, keyLen, plaintext);
}

static std::string shortToBytesLe(int16_t input) {
  int8_t output[2];
  memset(output, 0, 2);
  uint16_t in = static_cast<uint16_t>(input);
  output[1] = static_cast<int8_t>(0xff & (in >> 8));
  output[0] = static_cast<int8_t>(0xff & (in));

  return std::string(reinterpret_cast<char const*>(output), 2);
}

static void checkPageOrdinal(int32_t pageOrdinal) {
  if (ARROW_PREDICT_FALSE(pageOrdinal > std::numeric_limits<int16_t>::max())) {
    throw ParquetException(
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
    int32_t pageOrdinal) {
  checkPageOrdinal(pageOrdinal);
  const int16_t pageOrdinalShort = static_cast<int16_t>(pageOrdinal);
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
  std::string pageOrdinalBytes = shortToBytesLe(pageOrdinalShort);
  std::ostringstream out;
  out << fileAad << typeOrdinalBytesStr << rowGroupOrdinalBytes
      << columnOrdinalBytes << pageOrdinalBytes;
  return out.str();
}

std::string createFooterAad(const std::string& aadPrefixBytes) {
  return createModuleAad(
      aadPrefixBytes,
      kFooter,
      static_cast<int16_t>(-1),
      static_cast<int16_t>(-1),
      static_cast<int16_t>(-1));
}

// Update last two bytes with new page ordinal (instead of creating new page
// AAD. from scratch)
void quickUpdatePageAad(int32_t newPageOrdinal, std::string* AAD) {
  checkPageOrdinal(newPageOrdinal);
  const std::string pageOrdinalBytes =
      shortToBytesLe(static_cast<int16_t>(newPageOrdinal));
  std::memcpy(AAD->data() + AAD->length() - 2, pageOrdinalBytes.data(), 2);
}

void randBytes(unsigned char* buf, int num) {
  if (RAND_bytes(buf, num) != 1) {
    throw ParquetException("Failed to generate random bytes");
  }
}

} // namespace facebook::velox::parquet::arrow::encryption
