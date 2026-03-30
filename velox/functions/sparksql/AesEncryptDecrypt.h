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

#include "velox/common/Enums.h"
#include "velox/common/base/Status.h"
#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"

#include <folly/ScopeGuard.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

namespace facebook::velox::functions::sparksql {

namespace detail {

enum class CipherMode { ECB, CBC, GCM };

VELOX_DECLARE_ENUM_NAME(CipherMode);

struct CipherConfig {
  CipherMode mode;
  int ivLen{};
  bool usePkcs{};
  bool supportsAad{};
  std::string error;
};

inline CipherConfig parseModeAndPadding(
    const std::string& modeStr,
    const std::string& paddingStr) {
  CipherConfig config;

  if (modeStr == "ECB") {
    config.mode = CipherMode::ECB;
    config.ivLen = 0;
    config.supportsAad = false;
  } else if (modeStr == "CBC") {
    config.mode = CipherMode::CBC;
    config.ivLen = 16;
    config.supportsAad = false;
  } else if (modeStr == "GCM") {
    config.mode = CipherMode::GCM;
    config.ivLen = 12;
    config.supportsAad = true;
  } else {
    config.error = fmt::format("Unsupported AES mode: {}", modeStr);
    return config;
  }

  if (paddingStr == "DEFAULT") {
    config.usePkcs = (config.mode != CipherMode::GCM);
  } else if (paddingStr == "PKCS") {
    if (config.mode == CipherMode::GCM) {
      config.error = "PKCS padding is not supported for GCM mode";
      return config;
    }
    config.usePkcs = true;
  } else if (paddingStr == "NONE") {
    config.usePkcs = false;
  } else {
    config.error = fmt::format("Unsupported AES padding: {}", paddingStr);
    return config;
  }

  return config;
}

inline const EVP_CIPHER* getCipher(CipherMode mode, int keyLen) {
  switch (mode) {
    case CipherMode::ECB:
      switch (keyLen) {
        case 16:
          return EVP_aes_128_ecb();
        case 24:
          return EVP_aes_192_ecb();
        case 32:
          return EVP_aes_256_ecb();
      }
      break;
    case CipherMode::CBC:
      switch (keyLen) {
        case 16:
          return EVP_aes_128_cbc();
        case 24:
          return EVP_aes_192_cbc();
        case 32:
          return EVP_aes_256_cbc();
      }
      break;
    case CipherMode::GCM:
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

} // namespace detail

template <typename T>
struct AesEncryptFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void initialize(
      const std::vector<TypePtr>&,
      const core::QueryConfig&,
      const arg_type<Varbinary>* /*input*/,
      const arg_type<Varbinary>* /*key*/,
      const arg_type<Varchar>* mode,
      const arg_type<Varchar>* padding,
      const arg_type<Varbinary>* /*iv*/,
      const arg_type<Varbinary>* /*aad*/) {
    std::string modeStr = mode ? mode->str() : "GCM";
    std::string paddingStr = padding ? padding->str() : "DEFAULT";
    std::transform(modeStr.begin(), modeStr.end(), modeStr.begin(), ::toupper);
    std::transform(
        paddingStr.begin(), paddingStr.end(), paddingStr.begin(), ::toupper);
    config_ = detail::parseModeAndPadding(modeStr, paddingStr);
  }

  FOLLY_ALWAYS_INLINE Status callNullable(
      out_type<Varbinary>& result,
      const arg_type<Varbinary>* input,
      const arg_type<Varbinary>* key,
      const arg_type<Varchar>* /*mode*/,
      const arg_type<Varchar>* /*padding*/,
      const arg_type<Varbinary>* iv,
      const arg_type<Varbinary>* aad) {
    if (input == nullptr || key == nullptr) {
      return Status::OK();
    }
    if (!config_.error.empty()) {
      return Status::UserError(config_.error);
    }
    const StringView ivView =
        iv ? StringView(iv->data(), iv->size()) : StringView();
    const StringView aadView =
        aad ? StringView(aad->data(), aad->size()) : StringView();
    aesEncryptImpl(
        result,
        StringView(input->data(), input->size()),
        StringView(key->data(), key->size()),
        config_,
        ivView,
        aadView);
    return Status::OK();
  }

 private:
  detail::CipherConfig config_;

  template <typename TResult>
  static void aesEncryptImpl(
      TResult& result,
      const StringView& input,
      const StringView& key,
      const detail::CipherConfig& config,
      const StringView& iv,
      const StringView& aad) {
    const auto keyLen = static_cast<int>(key.size());
    VELOX_USER_CHECK(
        keyLen == 16 || keyLen == 24 || keyLen == 32,
        "Invalid AES key length: {}. Must be 16, 24, or 32 bytes.",
        keyLen);

    if (!config.usePkcs && config.mode != detail::CipherMode::GCM &&
        input.size() % 16 != 0) {
      VELOX_USER_FAIL(
          "Input size must be a multiple of 16 bytes for {} mode with NONE "
          "padding. Got {} bytes.",
          detail::CipherModeName::toName(config.mode),
          input.size());
    }

    const auto* cipher = detail::getCipher(config.mode, keyLen);
    const auto blockSize = EVP_CIPHER_block_size(cipher);
    const auto gcmTagLen = (config.mode == detail::CipherMode::GCM) ? 16 : 0;
    const auto maxSize =
        config.ivLen + static_cast<int>(input.size()) + blockSize + gcmTagLen;
    result.reserve(maxSize);

    // Write IV directly to result buffer.
    if (config.ivLen > 0) {
      if (iv.size() > 0) {
        VELOX_USER_CHECK_EQ(iv.size(), config.ivLen, "Invalid IV length");
        std::memcpy(result.data(), iv.data(), config.ivLen);
      } else {
        VELOX_USER_CHECK(
            RAND_bytes(
                reinterpret_cast<unsigned char*>(result.data()),
                config.ivLen) == 1,
            "Failed to generate random IV");
      }
    } else if (iv.size() > 0) {
      VELOX_USER_FAIL(
          "IV is not supported for {} mode",
          detail::CipherModeName::toName(config.mode));
    }

    if (aad.size() > 0 && !config.supportsAad) {
      VELOX_USER_FAIL(
          "AAD is not supported for {} mode",
          detail::CipherModeName::toName(config.mode));
    }

    auto* ctx = EVP_CIPHER_CTX_new();
    VELOX_CHECK_NOT_NULL(ctx);
    SCOPE_EXIT {
      EVP_CIPHER_CTX_free(ctx);
    };

    if (config.mode == detail::CipherMode::GCM) {
      // GCM requires two-step init: set cipher, set IV length, then key+IV.
      VELOX_USER_CHECK(
          EVP_EncryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr) == 1,
          "Failed to initialize AES encryption");
      VELOX_USER_CHECK(
          EVP_CIPHER_CTX_ctrl(
              ctx, EVP_CTRL_GCM_SET_IVLEN, config.ivLen, nullptr) == 1,
          "Failed to set GCM IV length");
      VELOX_USER_CHECK(
          EVP_EncryptInit_ex(
              ctx,
              nullptr,
              nullptr,
              reinterpret_cast<const unsigned char*>(key.data()),
              reinterpret_cast<const unsigned char*>(result.data())) == 1,
          "Failed to initialize AES encryption");
    } else {
      VELOX_USER_CHECK(
          EVP_EncryptInit_ex(
              ctx,
              cipher,
              nullptr,
              reinterpret_cast<const unsigned char*>(key.data()),
              config.ivLen > 0
                  ? reinterpret_cast<const unsigned char*>(result.data())
                  : nullptr) == 1,
          "Failed to initialize AES encryption");
    }

    if (!config.usePkcs) {
      EVP_CIPHER_CTX_set_padding(ctx, 0);
    }

    if (config.mode == detail::CipherMode::GCM && aad.size() > 0) {
      int aadLen = 0;
      VELOX_USER_CHECK(
          EVP_EncryptUpdate(
              ctx,
              nullptr,
              &aadLen,
              reinterpret_cast<const unsigned char*>(aad.data()),
              aad.size()) == 1,
          "Failed to process AAD");
    }

    int outLen = 0;
    auto* outPtr =
        reinterpret_cast<unsigned char*>(result.data()) + config.ivLen;
    VELOX_USER_CHECK(
        EVP_EncryptUpdate(
            ctx,
            outPtr,
            &outLen,
            reinterpret_cast<const unsigned char*>(input.data()),
            input.size()) == 1,
        "AES encryption failed");

    int finalLen = 0;
    VELOX_USER_CHECK(
        EVP_EncryptFinal_ex(ctx, outPtr + outLen, &finalLen) == 1,
        "AES encryption finalization failed");

    auto totalLen = outLen + finalLen;
    if (config.mode == detail::CipherMode::GCM) {
      VELOX_USER_CHECK(
          EVP_CIPHER_CTX_ctrl(
              ctx, EVP_CTRL_GCM_GET_TAG, 16, outPtr + totalLen) == 1,
          "Failed to get GCM tag");
      totalLen += 16;
    }

    VELOX_DCHECK_LE(config.ivLen + totalLen, maxSize);
    result.resize(config.ivLen + totalLen);
  }
};

template <typename T>
struct AesDecryptFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void initialize(
      const std::vector<TypePtr>&,
      const core::QueryConfig&,
      const arg_type<Varbinary>* /*input*/,
      const arg_type<Varbinary>* /*key*/,
      const arg_type<Varchar>* mode,
      const arg_type<Varchar>* padding,
      const arg_type<Varbinary>* /*iv*/,
      const arg_type<Varbinary>* /*aad*/) {
    std::string modeStr = mode ? mode->str() : "GCM";
    std::string paddingStr = padding ? padding->str() : "DEFAULT";
    std::transform(modeStr.begin(), modeStr.end(), modeStr.begin(), ::toupper);
    std::transform(
        paddingStr.begin(), paddingStr.end(), paddingStr.begin(), ::toupper);
    config_ = detail::parseModeAndPadding(modeStr, paddingStr);
  }

  FOLLY_ALWAYS_INLINE Status callNullable(
      out_type<Varbinary>& result,
      const arg_type<Varbinary>* input,
      const arg_type<Varbinary>* key,
      const arg_type<Varchar>* /*mode*/,
      const arg_type<Varchar>* /*padding*/,
      const arg_type<Varbinary>* iv,
      const arg_type<Varbinary>* aad) {
    if (input == nullptr || key == nullptr) {
      return Status::OK();
    }
    if (!config_.error.empty()) {
      return Status::UserError(config_.error);
    }
    const StringView ivView =
        iv ? StringView(iv->data(), iv->size()) : StringView();
    const StringView aadView =
        aad ? StringView(aad->data(), aad->size()) : StringView();
    aesDecryptImpl(
        result,
        StringView(input->data(), input->size()),
        StringView(key->data(), key->size()),
        config_,
        ivView,
        aadView);
    return Status::OK();
  }

 private:
  detail::CipherConfig config_;

  template <typename TResult>
  static void aesDecryptImpl(
      TResult& result,
      const StringView& input,
      const StringView& key,
      const detail::CipherConfig& config,
      const StringView& iv,
      const StringView& aad) {
    const auto keyLen = static_cast<int>(key.size());
    VELOX_USER_CHECK(
        keyLen == 16 || keyLen == 24 || keyLen == 32,
        "Invalid AES key length: {}. Must be 16, 24, or 32 bytes.",
        keyLen);

    const auto gcmTagLen = (config.mode == detail::CipherMode::GCM) ? 16 : 0;
    const unsigned char* ivPtr = nullptr;
    const unsigned char* cipherData;
    int cipherLen;

    if (config.ivLen > 0) {
      if (iv.size() > 0) {
        VELOX_USER_CHECK_EQ(iv.size(), config.ivLen, "Invalid IV length");
        ivPtr = reinterpret_cast<const unsigned char*>(iv.data());
        cipherData = reinterpret_cast<const unsigned char*>(input.data());
        cipherLen = input.size();
      } else {
        VELOX_USER_CHECK_GE(input.size(), config.ivLen, "Input too short");
        ivPtr = reinterpret_cast<const unsigned char*>(input.data());
        cipherData =
            reinterpret_cast<const unsigned char*>(input.data()) + config.ivLen;
        cipherLen = input.size() - config.ivLen;
      }
    } else {
      cipherData = reinterpret_cast<const unsigned char*>(input.data());
      cipherLen = input.size();
    }

    const unsigned char* tag = nullptr;
    if (config.mode == detail::CipherMode::GCM) {
      VELOX_USER_CHECK_GE(
          cipherLen, gcmTagLen, "Input too short — missing GCM tag");
      cipherLen -= gcmTagLen;
      tag = cipherData + cipherLen;
    }

    const auto* cipher = detail::getCipher(config.mode, keyLen);
    auto* ctx = EVP_CIPHER_CTX_new();
    VELOX_CHECK_NOT_NULL(ctx);
    SCOPE_EXIT {
      EVP_CIPHER_CTX_free(ctx);
    };

    if (config.mode == detail::CipherMode::GCM) {
      // GCM requires two-step init: set cipher, set IV length, then key+IV.
      VELOX_USER_CHECK(
          EVP_DecryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr) == 1,
          "Failed to initialize AES decryption");
      VELOX_USER_CHECK(
          EVP_CIPHER_CTX_ctrl(
              ctx, EVP_CTRL_GCM_SET_IVLEN, config.ivLen, nullptr) == 1,
          "Failed to set GCM IV length");
      VELOX_USER_CHECK(
          EVP_DecryptInit_ex(
              ctx,
              nullptr,
              nullptr,
              reinterpret_cast<const unsigned char*>(key.data()),
              ivPtr) == 1,
          "Failed to initialize AES decryption");
      VELOX_USER_CHECK(
          EVP_CIPHER_CTX_ctrl(
              ctx,
              EVP_CTRL_GCM_SET_TAG,
              gcmTagLen,
              const_cast<unsigned char*>(tag)) == 1,
          "Failed to set GCM tag");
    } else {
      VELOX_USER_CHECK(
          EVP_DecryptInit_ex(
              ctx,
              cipher,
              nullptr,
              reinterpret_cast<const unsigned char*>(key.data()),
              ivPtr) == 1,
          "Failed to initialize AES decryption");
    }

    if (!config.usePkcs) {
      EVP_CIPHER_CTX_set_padding(ctx, 0);
    }

    if (config.mode == detail::CipherMode::GCM && aad.size() > 0) {
      int aadLen = 0;
      VELOX_USER_CHECK(
          EVP_DecryptUpdate(
              ctx,
              nullptr,
              &aadLen,
              reinterpret_cast<const unsigned char*>(aad.data()),
              aad.size()) == 1,
          "Failed to process AAD");
    }

    const auto maxSize = cipherLen + EVP_CIPHER_block_size(cipher);
    result.reserve(maxSize);
    int outLen = 0;
    VELOX_USER_CHECK(
        EVP_DecryptUpdate(
            ctx,
            reinterpret_cast<unsigned char*>(result.data()),
            &outLen,
            cipherData,
            cipherLen) == 1,
        "AES decryption failed");

    int finalLen = 0;
    VELOX_USER_CHECK(
        EVP_DecryptFinal_ex(
            ctx,
            reinterpret_cast<unsigned char*>(result.data()) + outLen,
            &finalLen) == 1,
        "AES decryption failed — invalid key, padding, or authentication tag");

    VELOX_DCHECK_LE(outLen + finalLen, maxSize);
    result.resize(outLen + finalLen);
  }
};

} // namespace facebook::velox::functions::sparksql
