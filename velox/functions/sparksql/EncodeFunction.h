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

#include <cstdint>
#include <string>
#include <vector>

#include "velox/common/base/Status.h"
#include "velox/expression/StringWriter.h"
#include "velox/functions/Macros.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"

namespace facebook::velox::functions::sparksql {

namespace detail {

enum class CharsetType : uint8_t {
  kUtf8,
  kUsAscii,
  kIso8859_1,
  kUtf16,
  kUtf16BE,
  kUtf16LE,
  kUtf16LEWithBom,
  kUtf32,
  kUtf32BE,
  kUtf32LE,
  kLegacy,
  kUnsupported,
};

/// Resolves a charset name using Spark's configured charset scope.
CharsetType resolveCharset(const StringView& charset, bool legacyJavaCharsets);

/// Encodes input using the selected charset and Spark coding-error behavior.
Status encode(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    CharsetType type,
    bool legacyCodingErrorAction);

} // namespace detail

/// Encodes a VARCHAR into VARBINARY using Spark-compatible charset semantics.
template <typename T>
struct EncodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* charset) {
    const SparkQueryConfig sparkConfig{config};
    legacyJavaCharsets_ = sparkConfig.legacyJavaCharsets();
    legacyCodingErrorAction_ = sparkConfig.legacyCodingErrorAction();
    if (charset != nullptr) {
      charsetType_ = detail::resolveCharset(*charset, legacyJavaCharsets_);
      isConstantCharset_ = true;
    }
  }

  Status call(
      out_type<Varbinary>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& charset) {
    const auto type = isConstantCharset_
        ? charsetType_
        : detail::resolveCharset(charset, legacyJavaCharsets_);
    if (type == detail::CharsetType::kUnsupported) {
      return Status::UserError(
          "encode: unsupported charset '{}'",
          std::string(charset.data(), charset.size()));
    }
    return detail::encode(
        result, input, charset, type, legacyCodingErrorAction_);
  }

 private:
  // Stores the charset resolved from a constant argument.
  detail::CharsetType charsetType_{detail::CharsetType::kUnsupported};

  // Indicates whether the charset argument was constant.
  bool isConstantCharset_{false};

  // Enables the configured Java charset alias scope.
  bool legacyJavaCharsets_{false};

  // Replaces unmappable characters instead of returning an error.
  bool legacyCodingErrorAction_{false};
};

} // namespace facebook::velox::functions::sparksql
