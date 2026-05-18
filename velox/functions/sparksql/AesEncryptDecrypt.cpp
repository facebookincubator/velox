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

#include "velox/functions/sparksql/AesEncryptDecrypt.h"
#include "velox/common/EnumDefine.h"

#include <folly/ScopeGuard.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

namespace facebook::velox::functions::sparksql::detail {

namespace {
const auto& cipherModeNames() {
  static const folly::F14FastMap<CipherMode, std::string_view> kNames = {
      {CipherMode::kEcb, "ECB"},
      {CipherMode::kCbc, "CBC"},
      {CipherMode::kGcm, "GCM"},
  };
  return kNames;
}
} // namespace

VELOX_DEFINE_ENUM_NAME(CipherMode, cipherModeNames);

CipherConfig parseModeAndPadding(
    const std::string& modeStr,
    const std::string& paddingStr) {
  CipherConfig config;

  if (modeStr == "ECB") {
    config.mode = CipherMode::kEcb;
    config.ivLen = 0;
    config.supportsAad = false;
  } else if (modeStr == "CBC") {
    config.mode = CipherMode::kCbc;
    config.ivLen = 16;
    config.supportsAad = false;
  } else if (modeStr == "GCM") {
    config.mode = CipherMode::kGcm;
    config.ivLen = 12;
    config.supportsAad = true;
  } else {
    VELOX_USER_FAIL("Unsupported AES mode: {}", modeStr);
  }

  if (paddingStr == "DEFAULT") {
    config.usePkcs = (config.mode != CipherMode::kGcm);
  } else if (paddingStr == "PKCS") {
    VELOX_USER_CHECK(
        config.mode != CipherMode::kGcm,
        "PKCS padding is not supported for GCM mode");
    config.usePkcs = true;
  } else if (paddingStr == "NONE") {
    config.usePkcs = false;
  } else {
    VELOX_USER_FAIL("Unsupported AES padding: {}", paddingStr);
  }

  return config;
}

const EVP_CIPHER* getCipher(CipherMode mode, int keyLen) {
  switch (mode) {
    case CipherMode::kEcb:
      switch (keyLen) {
        case 16:
          return EVP_aes_128_ecb();
        case 24:
          return EVP_aes_192_ecb();
        case 32:
          return EVP_aes_256_ecb();
      }
      break;
    case CipherMode::kCbc:
      switch (keyLen) {
        case 16:
          return EVP_aes_128_cbc();
        case 24:
          return EVP_aes_192_cbc();
        case 32:
          return EVP_aes_256_cbc();
      }
      break;
    case CipherMode::kGcm:
      switch (keyLen) {
        case 16:
          return EVP_aes_128_gcm();
        case 24:
          return EVP_aes_192_gcm();
        case 32:
          return EVP_aes_256_gcm();
      }
      break;
  }
  VELOX_UNREACHABLE();
}

void aesEncryptImpl(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& key,
    const CipherConfig& config,
    const StringView* iv,
    const StringView* aad) {
  VELOX_USER_CHECK(
      fitsInInt(key.size()),
      "AES key size {} exceeds the maximum supported by OpenSSL EVP API "
      "(INT_MAX bytes)",
      key.size());
  VELOX_USER_CHECK(
      fitsInInt(input.size()),
      "AES input size {} exceeds the maximum supported by OpenSSL EVP API "
      "(INT_MAX bytes)",
      input.size());
  if (aad != nullptr) {
    VELOX_USER_CHECK(
        fitsInInt(aad->size()),
        "AES AAD size {} exceeds the maximum supported by OpenSSL EVP API "
        "(INT_MAX bytes)",
        aad->size());
  }

  const auto keyLen = static_cast<int>(key.size());
  VELOX_USER_CHECK(
      keyLen == 16 || keyLen == 24 || keyLen == 32,
      "Invalid AES key length: {}. Must be 16, 24, or 32 bytes.",
      keyLen);

  if (!config.usePkcs && config.mode != CipherMode::kGcm &&
      input.size() % 16 != 0) {
    VELOX_USER_FAIL(
        "Input size must be a multiple of 16 bytes for {} mode with NONE "
        "padding. Got {} bytes.",
        CipherModeName::toName(config.mode),
        input.size());
  }

  const auto* cipher = getCipher(config.mode, keyLen);
  const auto blockSize = EVP_CIPHER_block_size(cipher);
  const auto gcmTagLen = (config.mode == CipherMode::kGcm) ? 16 : 0;
  // All four addends are bounded ints (input.size() already checked above,
  // ivLen/blockSize/gcmTagLen are tiny constants), so size_t arithmetic
  // here cannot overflow before the cast.
  const size_t maxSize = static_cast<size_t>(config.ivLen) + input.size() +
      static_cast<size_t>(blockSize) + static_cast<size_t>(gcmTagLen);
  VELOX_USER_CHECK(
      fitsInInt(maxSize),
      "AES output size {} exceeds the maximum supported by OpenSSL EVP API "
      "(INT_MAX bytes)",
      maxSize);
  result.reserve(maxSize);

  // Write IV directly to result buffer.
  if (config.ivLen > 0) {
    if (iv != nullptr && iv->size() > 0) {
      VELOX_USER_CHECK_EQ(
          static_cast<int>(iv->size()), config.ivLen, "Invalid IV length");
      std::memcpy(result.data(), iv->data(), config.ivLen);
    } else {
      VELOX_CHECK_EQ(
          RAND_bytes(
              reinterpret_cast<unsigned char*>(result.data()), config.ivLen),
          1,
          "Failed to generate random IV");
    }
  } else if (iv != nullptr && iv->size() > 0) {
    VELOX_USER_FAIL(
        "IV is not supported for {} mode", CipherModeName::toName(config.mode));
  }

  if (aad != nullptr && aad->size() > 0 && !config.supportsAad) {
    VELOX_USER_FAIL(
        "AAD is not supported for {} mode",
        CipherModeName::toName(config.mode));
  }

  auto* ctx = EVP_CIPHER_CTX_new();
  VELOX_CHECK_NOT_NULL(ctx);
  SCOPE_EXIT {
    EVP_CIPHER_CTX_free(ctx);
  };

  if (config.mode == CipherMode::kGcm) {
    // GCM requires two-step init: set cipher, set IV length, then key+IV.
    VELOX_CHECK_EQ(
        EVP_EncryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr),
        1,
        "Failed to initialize AES encryption");
    VELOX_CHECK_EQ(
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, config.ivLen, nullptr),
        1,
        "Failed to set GCM IV length");
    VELOX_CHECK_EQ(
        EVP_EncryptInit_ex(
            ctx,
            nullptr,
            nullptr,
            reinterpret_cast<const unsigned char*>(key.data()),
            reinterpret_cast<const unsigned char*>(result.data())),
        1,
        "Failed to initialize AES encryption");
  } else {
    VELOX_CHECK_EQ(
        EVP_EncryptInit_ex(
            ctx,
            cipher,
            nullptr,
            reinterpret_cast<const unsigned char*>(key.data()),
            config.ivLen > 0
                ? reinterpret_cast<const unsigned char*>(result.data())
                : nullptr),
        1,
        "Failed to initialize AES encryption");
  }

  if (!config.usePkcs) {
    EVP_CIPHER_CTX_set_padding(ctx, 0);
  }

  if (config.mode == CipherMode::kGcm && aad != nullptr && aad->size() > 0) {
    int aadLen = 0;
    VELOX_CHECK_EQ(
        EVP_EncryptUpdate(
            ctx,
            nullptr,
            &aadLen,
            reinterpret_cast<const unsigned char*>(aad->data()),
            static_cast<int>(aad->size())),
        1,
        "Failed to process AAD");
  }

  int outLen = 0;
  auto* outPtr = reinterpret_cast<unsigned char*>(result.data()) + config.ivLen;
  VELOX_CHECK_EQ(
      EVP_EncryptUpdate(
          ctx,
          outPtr,
          &outLen,
          reinterpret_cast<const unsigned char*>(input.data()),
          static_cast<int>(input.size())),
      1,
      "AES encryption failed");

  int finalLen = 0;
  VELOX_CHECK_EQ(
      EVP_EncryptFinal_ex(ctx, outPtr + outLen, &finalLen),
      1,
      "AES encryption finalization failed");

  auto totalLen = outLen + finalLen;
  if (config.mode == CipherMode::kGcm) {
    VELOX_CHECK_EQ(
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, outPtr + totalLen),
        1,
        "Failed to get GCM tag");
    totalLen += 16;
  }

  VELOX_DCHECK_LE(config.ivLen + totalLen, maxSize);
  result.resize(config.ivLen + totalLen);
}

void aesDecryptImpl(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& key,
    const CipherConfig& config,
    const StringView* iv,
    const StringView* aad) {
  VELOX_USER_CHECK(
      fitsInInt(key.size()),
      "AES key size {} exceeds the maximum supported by OpenSSL EVP API "
      "(INT_MAX bytes)",
      key.size());
  VELOX_USER_CHECK(
      fitsInInt(input.size()),
      "AES input size {} exceeds the maximum supported by OpenSSL EVP API "
      "(INT_MAX bytes)",
      input.size());
  if (aad != nullptr) {
    VELOX_USER_CHECK(
        fitsInInt(aad->size()),
        "AES AAD size {} exceeds the maximum supported by OpenSSL EVP API "
        "(INT_MAX bytes)",
        aad->size());
  }

  const auto keyLen = static_cast<int>(key.size());
  VELOX_USER_CHECK(
      keyLen == 16 || keyLen == 24 || keyLen == 32,
      "Invalid AES key length: {}. Must be 16, 24, or 32 bytes.",
      keyLen);

  const auto gcmTagLen = (config.mode == CipherMode::kGcm) ? 16 : 0;
  const unsigned char* ivPtr = nullptr;
  const unsigned char* cipherData;
  int cipherLen;

  if (config.ivLen > 0) {
    if (iv != nullptr && iv->size() > 0) {
      VELOX_USER_CHECK_EQ(
          static_cast<int>(iv->size()), config.ivLen, "Invalid IV length");
      ivPtr = reinterpret_cast<const unsigned char*>(iv->data());
      cipherData = reinterpret_cast<const unsigned char*>(input.data());
      cipherLen = static_cast<int>(input.size());
    } else {
      VELOX_USER_CHECK_GE(
          static_cast<int>(input.size()), config.ivLen, "Input too short");
      ivPtr = reinterpret_cast<const unsigned char*>(input.data());
      cipherData =
          reinterpret_cast<const unsigned char*>(input.data()) + config.ivLen;
      cipherLen = static_cast<int>(input.size()) - config.ivLen;
    }
  } else {
    cipherData = reinterpret_cast<const unsigned char*>(input.data());
    cipherLen = static_cast<int>(input.size());
  }

  const unsigned char* tag = nullptr;
  if (config.mode == CipherMode::kGcm) {
    VELOX_USER_CHECK_GE(
        cipherLen, gcmTagLen, "Input too short — missing GCM tag");
    cipherLen -= gcmTagLen;
    tag = cipherData + cipherLen;
  }

  const auto* cipher = getCipher(config.mode, keyLen);
  auto* ctx = EVP_CIPHER_CTX_new();
  VELOX_CHECK_NOT_NULL(ctx);
  SCOPE_EXIT {
    EVP_CIPHER_CTX_free(ctx);
  };

  if (config.mode == CipherMode::kGcm) {
    // GCM requires two-step init: set cipher, set IV length, then key+IV.
    VELOX_CHECK_EQ(
        EVP_DecryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr),
        1,
        "Failed to initialize AES decryption");
    VELOX_CHECK_EQ(
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, config.ivLen, nullptr),
        1,
        "Failed to set GCM IV length");
    VELOX_CHECK_EQ(
        EVP_DecryptInit_ex(
            ctx,
            nullptr,
            nullptr,
            reinterpret_cast<const unsigned char*>(key.data()),
            ivPtr),
        1,
        "Failed to initialize AES decryption");
    VELOX_CHECK_EQ(
        EVP_CIPHER_CTX_ctrl(
            ctx,
            EVP_CTRL_GCM_SET_TAG,
            gcmTagLen,
            const_cast<unsigned char*>(tag)),
        1,
        "Failed to set GCM tag");
  } else {
    VELOX_CHECK_EQ(
        EVP_DecryptInit_ex(
            ctx,
            cipher,
            nullptr,
            reinterpret_cast<const unsigned char*>(key.data()),
            ivPtr),
        1,
        "Failed to initialize AES decryption");
  }

  if (!config.usePkcs) {
    EVP_CIPHER_CTX_set_padding(ctx, 0);
  }

  if (config.mode == CipherMode::kGcm && aad != nullptr && aad->size() > 0) {
    int aadLen = 0;
    VELOX_CHECK_EQ(
        EVP_DecryptUpdate(
            ctx,
            nullptr,
            &aadLen,
            reinterpret_cast<const unsigned char*>(aad->data()),
            static_cast<int>(aad->size())),
        1,
        "Failed to process AAD");
  }

  const auto maxSize = cipherLen + EVP_CIPHER_block_size(cipher);
  result.reserve(maxSize);
  int outLen = 0;
  VELOX_CHECK_EQ(
      EVP_DecryptUpdate(
          ctx,
          reinterpret_cast<unsigned char*>(result.data()),
          &outLen,
          cipherData,
          cipherLen),
      1,
      "AES decryption failed");

  int finalLen = 0;
  VELOX_USER_CHECK_EQ(
      EVP_DecryptFinal_ex(
          ctx,
          reinterpret_cast<unsigned char*>(result.data()) + outLen,
          &finalLen),
      1,
      "AES decryption failed — invalid key, padding, or authentication tag");

  VELOX_DCHECK_LE(outLen + finalLen, maxSize);
  result.resize(outLen + finalLen);
}

} // namespace facebook::velox::functions::sparksql::detail
