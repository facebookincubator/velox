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

// Returns true if a size_t value safely fits into a non-negative `int`,
// i.e. the OpenSSL EVP_*Update length parameter type.
FOLLY_ALWAYS_INLINE bool fitsInInt(size_t n) {
  return n <= static_cast<size_t>(std::numeric_limits<int>::max());
}

// Implementations live in AesEncryptDecrypt.cpp — heavy OpenSSL logic that
// should not be in the header. iv/aad are pointers because they are optional
// arguments (nullptr means absent).
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

  // Returns false to signal a NULL output when input or key is NULL.
  // Returning Status with notNull=true implicitly is what `Status` callNullable
  // does (see SimpleFunctionMetadata.h:1144), so we use bool to retain control
  // over the null bit. Errors are surfaced via VELOX_USER_FAIL since bool
  // callNullable cannot return a Status.
  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Varbinary>& result,
      const arg_type<Varbinary>* input,
      const arg_type<Varbinary>* key,
      const arg_type<Varchar>* /*mode*/,
      const arg_type<Varchar>* /*padding*/,
      const arg_type<Varbinary>* iv,
      const arg_type<Varbinary>* aad) {
    if (input == nullptr || key == nullptr) {
      return false;
    }
    detail::aesEncryptImpl(result, *input, *key, config_, iv, aad);
    return true;
  }

 private:
  detail::CipherConfig config_;
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

  // See AesEncryptFunction::callNullable comment for the bool-vs-Status
  // rationale.
  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Varbinary>& result,
      const arg_type<Varbinary>* input,
      const arg_type<Varbinary>* key,
      const arg_type<Varchar>* /*mode*/,
      const arg_type<Varchar>* /*padding*/,
      const arg_type<Varbinary>* iv,
      const arg_type<Varbinary>* aad) {
    if (input == nullptr || key == nullptr) {
      return false;
    }
    detail::aesDecryptImpl(result, *input, *key, config_, iv, aad);
    return true;
  }

 private:
  detail::CipherConfig config_;
};

} // namespace facebook::velox::functions::sparksql
