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

#include "velox/common/EnumDeclare.h"
#include "velox/common/base/Exceptions.h"
#include "velox/core/QueryConfig.h"
#include "velox/expression/StringWriter.h"
#include "velox/functions/Macros.h"

#include <limits>
#include <optional>

namespace facebook::velox::functions::sparksql {

namespace detail {

enum class CipherMode {
  kEcb,
  kCbc,
  kGcm,
};

VELOX_DECLARE_ENUM_NAME(CipherMode);

struct CipherConfig {
  CipherMode mode{CipherMode::kGcm};
  int ivLen{0};
  bool usePkcs{false};
  bool supportsAad{false};
};

CipherConfig parseModeAndPadding(
    const std::string& modeStr,
    const std::string& paddingStr);

FOLLY_ALWAYS_INLINE bool fitsInInt(size_t n) {
  return n <= static_cast<size_t>(std::numeric_limits<int>::max());
}

void aesEncryptImpl(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& key,
    const CipherConfig& config,
    const StringView* iv,
    const StringView* aad);

void aesDecryptImpl(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& key,
    const CipherConfig& config,
    const StringView* iv,
    const StringView* aad);

// Resolve mode/padding to a CipherConfig: prefer the cached value when
// mode/padding were literal at initialize time; otherwise parse the per-row
// strings. Returns nullopt when either mode or padding is null at runtime
// (Spark vanilla behavior: null mode/padding produces null output).
inline std::optional<CipherConfig> resolveCipherConfig(
    const std::optional<CipherConfig>& cached,
    const StringView* mode,
    const StringView* padding) {
  if (cached.has_value()) {
    return cached;
  }
  if (mode == nullptr || padding == nullptr) {
    return std::nullopt;
  }
  std::string modeStr = std::string(mode->data(), mode->size());
  std::string paddingStr = std::string(padding->data(), padding->size());
  std::transform(modeStr.begin(), modeStr.end(), modeStr.begin(), ::toupper);
  std::transform(
      paddingStr.begin(), paddingStr.end(), paddingStr.begin(), ::toupper);
  return parseModeAndPadding(modeStr, paddingStr);
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
    if (mode != nullptr && padding != nullptr) {
      std::string modeStr = mode->str();
      std::string paddingStr = padding->str();
      std::transform(
          modeStr.begin(), modeStr.end(), modeStr.begin(), ::toupper);
      std::transform(
          paddingStr.begin(), paddingStr.end(), paddingStr.begin(), ::toupper);
      // Swallow parse errors here so initialize never throws. Bad literal
      // mode/padding must throw from callNullable instead — otherwise the
      // expression fuzzer flags a divergence between common and simplified
      // eval paths (one throws at initialize, the other at eval).
      try {
        cachedConfig_ = detail::parseModeAndPadding(modeStr, paddingStr);
      } catch (const VeloxUserError&) {
      }
    }
  }

  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Varbinary>& result,
      const arg_type<Varbinary>* input,
      const arg_type<Varbinary>* key,
      const arg_type<Varchar>* mode,
      const arg_type<Varchar>* padding,
      const arg_type<Varbinary>* iv,
      const arg_type<Varbinary>* aad) {
    if (input == nullptr || key == nullptr) {
      return false;
    }
    auto cfg = detail::resolveCipherConfig(cachedConfig_, mode, padding);
    if (!cfg.has_value()) {
      return false;
    }
    detail::aesEncryptImpl(result, *input, *key, *cfg, iv, aad);
    return true;
  }

 private:
  std::optional<detail::CipherConfig> cachedConfig_;
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
    if (mode != nullptr && padding != nullptr) {
      std::string modeStr = mode->str();
      std::string paddingStr = padding->str();
      std::transform(
          modeStr.begin(), modeStr.end(), modeStr.begin(), ::toupper);
      std::transform(
          paddingStr.begin(), paddingStr.end(), paddingStr.begin(), ::toupper);
      // See AesEncryptFunction::initialize — must not throw here.
      try {
        cachedConfig_ = detail::parseModeAndPadding(modeStr, paddingStr);
      } catch (const VeloxUserError&) {
      }
    }
  }

  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Varbinary>& result,
      const arg_type<Varbinary>* input,
      const arg_type<Varbinary>* key,
      const arg_type<Varchar>* mode,
      const arg_type<Varchar>* padding,
      const arg_type<Varbinary>* iv,
      const arg_type<Varbinary>* aad) {
    if (input == nullptr || key == nullptr) {
      return false;
    }
    auto cfg = detail::resolveCipherConfig(cachedConfig_, mode, padding);
    if (!cfg.has_value()) {
      return false;
    }
    detail::aesDecryptImpl(result, *input, *key, *cfg, iv, aad);
    return true;
  }

 private:
  std::optional<detail::CipherConfig> cachedConfig_;
};

} // namespace facebook::velox::functions::sparksql
