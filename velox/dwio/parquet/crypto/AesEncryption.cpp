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

constexpr int kGcmMode = 0;
constexpr int kCtrMode = 1;
constexpr int kCtrIvLength = 16;

#define ENCRYPT_INIT(CTX, ALG)                                        \
  if (1 != EVP_EncryptInit_ex(CTX, ALG, nullptr, nullptr, nullptr)) { \
    throw CryptoException("Couldn't init ALG encryption");           \
  }

#define DECRYPT_INIT(CTX, ALG)                                        \
  if (1 != EVP_DecryptInit_ex(CTX, ALG, nullptr, nullptr, nullptr)) { \
    throw CryptoException("Couldn't init ALG decryption");           \
  }

class AesDecryptor::AesDecryptorImpl {
 public:
  explicit AesDecryptorImpl(
      ParquetCipher::type algId,
      int keyLen,
      bool metadata,
      bool containsLength);

  ~AesDecryptorImpl() {
    wipeOut();
  }

  int decrypt(
      const uint8_t* ciphertext,
      int ciphertextLen,
      const uint8_t* key,
      int keyLen,
      const uint8_t* aad,
      int aadLen,
      uint8_t* plaintext,
      int plaintextLen);

  void wipeOut() {
    if (nullptr != ctx_) {
      EVP_CIPHER_CTX_free(ctx_);
      ctx_ = nullptr;
    }
  }

  [[nodiscard]] int plaintextLength(int ciphertextLen) const {
    if (ciphertextLen < ciphertextSizeDelta_) {
      std::stringstream ss;
      ss << "Ciphertext length " << ciphertextLen
         << " is invalid, expected at least " << ciphertextSizeDelta_;
      throw CryptoException(ss.str());
    }
    return ciphertextLen - ciphertextSizeDelta_;
  }

  [[nodiscard]] int ciphertextLength(int plaintextLen) const {
    if (plaintextLen < 0) {
      std::stringstream ss;
      ss << "Negative plaintext length " << plaintextLen;
      throw CryptoException(ss.str());
    }
    return plaintextLen + ciphertextSizeDelta_;
  }

  /// Get the actual ciphertext length, inclusive of the length buffer length,
  /// and validate that the provided buffer size is large enough.
  [[nodiscard]] int getCiphertextLength(
      const uint8_t* ciphertext,
      int ciphertextLen) const;

  /// Get the actual ciphertext length, inclusive of the length buffer length, without validation.
  [[nodiscard]] int getCiphertextLengthWithoutValidation(
      const uint8_t* ciphertext,
      int ciphertextLen) const;

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
      uint8_t* plaintext,
      int plaintextLen);

  int ctrDecrypt(
      const uint8_t* ciphertext,
      int ciphertextLen,
      const uint8_t* key,
      int keyLen,
      uint8_t* plaintext,
      int plaintextLen);
};

int AesDecryptor::decrypt(
    const uint8_t* ciphertext,
    int ciphertextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    uint8_t* plaintext,
    int plaintextLen) {
  return impl_->decrypt(ciphertext, ciphertextLen, key, keyLen, aad, aadLen, plaintext, plaintextLen);
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
  if (metadata || (ParquetCipher::AES_GCM_V1 == algId)) {
    aesMode_ = kGcmMode;
    ciphertextSizeDelta_ += kGcmTagLength;
  } else {
    aesMode_ = kCtrMode;
  }

  if (16 != keyLen && 24 != keyLen && 32 != keyLen) {
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
    if (16 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_128_gcm());
    } else if (24 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_192_gcm());
    } else if (32 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_256_gcm());
    }
  } else {
    // Init AES-CTR with specified key length
    if (16 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_128_ctr());
    } else if (24 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_192_ctr());
    } else if (32 == keyLen) {
      DECRYPT_INIT(ctx_, EVP_aes_256_ctr());
    }
  }
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
    bool metadata) {
  if (ParquetCipher::AES_GCM_V1 != algId &&
      ParquetCipher::AES_GCM_CTR_V1 != algId) {
    std::stringstream ss;
    ss << "Crypto algorithm " << algId << " is not supported";
    throw CryptoException(ss.str());
  }

  auto decryptor = std::make_shared<AesDecryptor>(algId, keyLen, metadata);
  return decryptor;
}

int AesDecryptor::plaintextLength(int ciphertextLen) const {
  return impl_->plaintextLength(ciphertextLen);
}

int AesDecryptor::ciphertextLength(int plaintextLen) const {
  return impl_->ciphertextLength(plaintextLen);
}

int AesDecryptor::getCiphertextLength(const uint8_t* ciphertext, int ciphertextLen) const {
  return impl_->getCiphertextLength(ciphertext, ciphertextLen);
}

int AesDecryptor::getCiphertextLengthWithoutValidation(const uint8_t* ciphertext, int ciphertextLen) const {
  return impl_->getCiphertextLengthWithoutValidation(ciphertext, ciphertextLen);
}

int AesDecryptor::AesDecryptorImpl::getCiphertextLength(
    const uint8_t* ciphertext,
    int ciphertextLen) const {
  if (lengthBufferLength_ > 0) {
    // Note: length_buffer_length_ must be either 0 or kBufferSizeLength
    if (ciphertextLen < kBufferSizeLength) {
      std::stringstream ss;
      ss << "Ciphertext buffer length " << ciphertextLen
         << " is insufficient to read the ciphertext length." << " At least "
         << kBufferSizeLength << " bytes are required.";
      throw CryptoException(ss.str());
    }

    // Extract ciphertext length
    int written_ciphertext_len = ((ciphertext[3] & 0xff) << 24) |
        ((ciphertext[2] & 0xff) << 16) | ((ciphertext[1] & 0xff) << 8) |
        ((ciphertext[0] & 0xff));

    if (written_ciphertext_len < 0) {
      std::stringstream ss;
      ss << "Negative ciphertext length " << written_ciphertext_len;
      throw CryptoException(ss.str());
    } else if (
        ciphertextLen <
        written_ciphertext_len + lengthBufferLength_) {
      std::stringstream ss;
      ss << "Serialized ciphertext length "
         << (written_ciphertext_len + lengthBufferLength_)
         << " is greater than the provided ciphertext buffer length "
         << ciphertextLen;
      throw CryptoException(ss.str());
    }

    return written_ciphertext_len + lengthBufferLength_;
  } else {
    if (ciphertextLen >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      std::stringstream ss;
      ss << "Ciphertext buffer length " << ciphertextLen << " overflows int32";
      throw CryptoException(ss.str());
    }
    return static_cast<int>(ciphertextLen);
  }
}

int AesDecryptor::AesDecryptorImpl::getCiphertextLengthWithoutValidation(
    const uint8_t* ciphertext,
    int ciphertextLen) const {
  if (lengthBufferLength_ > 0) {
    // Note: length_buffer_length_ must be either 0 or kBufferSizeLength
    if (ciphertextLen < kBufferSizeLength) {
      std::stringstream ss;
      ss << "Ciphertext buffer length " << ciphertextLen
         << " is insufficient to read the ciphertext length." << " At least "
         << kBufferSizeLength << " bytes are required.";
      throw CryptoException(ss.str());
    }

    // Extract ciphertext length
    int written_ciphertext_len = ((ciphertext[3] & 0xff) << 24) |
        ((ciphertext[2] & 0xff) << 16) | ((ciphertext[1] & 0xff) << 8) |
        ((ciphertext[0] & 0xff));

    if (written_ciphertext_len < 0) {
      std::stringstream ss;
      ss << "Negative ciphertext length " << written_ciphertext_len;
      throw CryptoException(ss.str());
    }

    return written_ciphertext_len + lengthBufferLength_;
  } else {
    if (ciphertextLen >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      std::stringstream ss;
      ss << "Ciphertext buffer length " << ciphertextLen << " overflows int32";
      throw CryptoException(ss.str());
    }
    return static_cast<int>(ciphertextLen);
  }
}

int AesDecryptor::AesDecryptorImpl::gcmDecrypt(
    const uint8_t* ciphertext,
    int ciphertextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    uint8_t* plaintext,
    int plaintextLen) {
  int len;
  int plaintext_len;

  uint8_t tag[kGcmTagLength];
  memset(tag, 0, kGcmTagLength);
  uint8_t nonce[kNonceLength];
  memset(nonce, 0, kNonceLength);

  int ciphertext_len = getCiphertextLength(ciphertext, ciphertextLen);

  if (plaintextLen < ciphertext_len - ciphertextSizeDelta_) {
    std::stringstream ss;
    ss << "Plaintext buffer length " << plaintextLen << " is insufficient "
       << "for ciphertext length " << ciphertext_len;
    throw CryptoException(ss.str());
  }

  if (ciphertext_len < lengthBufferLength_ + kNonceLength + kGcmTagLength) {
    std::stringstream ss;
    ss << "Invalid ciphertext length " << ciphertext_len
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
      ciphertext + ciphertext_len - kGcmTagLength,
      ciphertext + ciphertext_len,
      tag);

  // Setting key and IV
  if (1 != EVP_DecryptInit_ex(ctx_, nullptr, nullptr, key, nonce)) {
    throw CryptoException("Couldn't set key and IV");
  }

  // Setting additional authenticated data
  if (aad && aadLen > 0 &&
      (1 !=
       EVP_DecryptUpdate(ctx_, nullptr, &len, aad, aadLen))) {
    throw CryptoException("Couldn't set AAD");
  }

  // Decryption
  if (!EVP_DecryptUpdate(
          ctx_,
          plaintext,
          &len,
          ciphertext + lengthBufferLength_ + kNonceLength,
          ciphertext_len - lengthBufferLength_ - kNonceLength -
              kGcmTagLength)) {
    throw CryptoException("Failed decryption update gcm");
  }

  plaintext_len = len;

  // Checking the tag (authentication)
  if (!EVP_CIPHER_CTX_ctrl(ctx_, EVP_CTRL_GCM_SET_TAG, kGcmTagLength, tag)) {
    throw CryptoException("Failed authentication");
  }

  // Finalization
  if (1 != EVP_DecryptFinal_ex(ctx_, plaintext + len, &len)) {
    throw CryptoException("Failed decryption finalization gcm");
  }

  plaintext_len += len;
  return plaintext_len;
}

int AesDecryptor::AesDecryptorImpl::ctrDecrypt(
    const uint8_t* ciphertext,
    int ciphertextLen,
    const uint8_t* key,
    int keyLen,
    uint8_t* plaintext,
    int plaintextLen) {
  int len;
  int plaintext_len;

  uint8_t iv[kCtrIvLength];
  memset(iv, 0, kCtrIvLength);

  int ciphertext_len = getCiphertextLength(ciphertext, ciphertextLen);

  if (plaintextLen < ciphertextLen - ciphertextSizeDelta_) {
    std::stringstream ss;
    ss << "Plaintext buffer length " << plaintextLen << " is insufficient "
       << "for ciphertext length " << ciphertext_len;
    throw CryptoException(ss.str());
  }

  if (ciphertext_len < lengthBufferLength_ + kNonceLength) {
    std::stringstream ss;
    ss << "Invalid ciphertext length " << ciphertext_len
       << ". Expected at least " << lengthBufferLength_ + kNonceLength
       << "\n";
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
          ciphertext_len - lengthBufferLength_ - kNonceLength)) {
    throw CryptoException("Failed decryption update ctr");
  }

  plaintext_len = len;

  // Finalization
  if (1 != EVP_DecryptFinal_ex(ctx_, plaintext + len, &len)) {
    throw CryptoException("Failed decryption finalization ctr");
  }

  plaintext_len += len;
  return plaintext_len;
}

int AesDecryptor::AesDecryptorImpl::decrypt(
    const uint8_t* ciphertext,
    int ciphertextLen,
    const uint8_t* key,
    int keyLen,
    const uint8_t* aad,
    int aadLen,
    uint8_t* plaintext,
    int plaintextLen) {
  if (static_cast<size_t>(keyLength_) != keyLen) {
    std::stringstream ss;
    ss << "Wrong key length " << keyLen << ". Should be " << keyLength_;
    throw CryptoException(ss.str());
  }

  if (kGcmMode == aesMode_) {
    return gcmDecrypt(
        ciphertext,
        ciphertextLen,
        key,
        keyLen,
        aad,
        aadLen,
        plaintext,
        plaintextLen);
  }

  return ctrDecrypt(
      ciphertext, ciphertextLen, key, keyLen, plaintext, plaintextLen);
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

 std::string createModuleAad(const std::string& fileAad, int8_t moduleType,
                             int16_t rowGroupOrdinal, int16_t columnOrdinal, int16_t pageOrdinal) {
   CheckPageOrdinal(pageOrdinal);
   int8_t type_ordinal_bytes[1];
   type_ordinal_bytes[0] = moduleType;
   std::string type_ordinal_bytes_str(reinterpret_cast<char const*>(type_ordinal_bytes), 1);
   if (kFooter == moduleType) {
     std::string result = fileAad + type_ordinal_bytes_str;
     return result;
   }
   std::string row_group_ordinal_bytes = shortToBytesLe(rowGroupOrdinal);
   std::string column_ordinal_bytes = shortToBytesLe(columnOrdinal);
   if (kDataPage != moduleType && kDataPageHeader != moduleType) {
     std::ostringstream out;
     out << fileAad << type_ordinal_bytes_str << row_group_ordinal_bytes
         << column_ordinal_bytes;
     return out.str();
   }
   std::string page_ordinal_bytes = shortToBytesLe(pageOrdinal);
   std::ostringstream out;
   out << fileAad << type_ordinal_bytes_str << row_group_ordinal_bytes
       << column_ordinal_bytes << page_ordinal_bytes;
   return out.str();
 }

}
