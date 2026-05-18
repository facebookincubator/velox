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

} // namespace facebook::velox::functions::sparksql::detail
