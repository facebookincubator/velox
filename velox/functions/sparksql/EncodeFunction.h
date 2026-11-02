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
  kUtf32,
  kLegacySingleByte,
  kUnsupported,
};

/// Identifies a standard charset or an entry in the verified single-byte table.
struct Charset {
  /// Selects a standard encoder or the verified single-byte table.
  CharsetType type{CharsetType::kUnsupported};
  /// Indexes the table only when type is kLegacySingleByte.
  uint8_t legacyIndex{0};
};

/// Resolves standard names, optionally including verified Java names and
/// aliases.
Charset resolveCharset(const StringView& charset, bool legacyJavaCharsets);

/// Encodes input using the selected charset and Spark coding-error behavior.
/// May borrow input bytes; callers must retain their backing storage.
Status encode(
    exec::StringWriter& result,
    const StringView& input,
    const StringView& charset,
    Charset resolved,
    bool legacyCodingErrorAction);

} // namespace detail

/// Encodes a VARCHAR into VARBINARY using Spark-compatible charset semantics.
template <typename T>
struct EncodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Keeps borrowed output bytes alive after the input vector is released.
  // The adapter retains input buffers even for conversion-only batches.
  static constexpr int32_t reuse_strings_from_arg = 0;

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* charset) {
    const SparkQueryConfig sparkConfig{config};
    legacyCodingErrorAction_ = sparkConfig.legacyCodingErrorAction();
    legacyJavaCharsets_ = sparkConfig.legacyJavaCharsets();
    if (charset != nullptr) {
      charset_ = detail::resolveCharset(*charset, legacyJavaCharsets_);
      isConstantCharset_ = true;
    }
  }

  Status call(
      out_type<Varbinary>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& charset) {
    auto resolved = charset_;
    // Avoid copying cheap standard names when the Java allowlist is disabled.
    if (!isConstantCharset_ && !legacyJavaCharsets_) {
      resolved = detail::resolveCharset(charset, false);
    } else if (
        !isConstantCharset_ &&
        (resolved.type == detail::CharsetType::kUnsupported ||
         StringView(cachedCharsetName_) != charset)) {
      resolved = detail::resolveCharset(charset, legacyJavaCharsets_);
      // Only supported names have a bounded length. Do not retain arbitrary
      // unsupported input strings in the function instance.
      if (resolved.type != detail::CharsetType::kUnsupported) {
        cachedCharsetName_.assign(charset.data(), charset.size());
        charset_ = resolved;
      }
    }
    if (resolved.type == detail::CharsetType::kUnsupported) {
      return Status::UserError(
          "encode: unsupported charset '{}'",
          std::string(charset.data(), charset.size()));
    }
    return detail::encode(
        result, input, charset, resolved, legacyCodingErrorAction_);
  }

 private:
  // Stores the constant charset or the last supported opt-in charset.
  detail::Charset charset_;

  // Owns the cached name because input vectors need not survive the next batch.
  std::string cachedCharsetName_;

  // Indicates whether the charset argument was constant.
  bool isConstantCharset_{false};

  // Replaces unmappable characters instead of returning an error.
  bool legacyCodingErrorAction_{false};

  // Enables only the explicitly verified Java names and single-byte encoders.
  bool legacyJavaCharsets_{false};
};

} // namespace facebook::velox::functions::sparksql
