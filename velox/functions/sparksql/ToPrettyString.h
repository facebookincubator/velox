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

#include <fmt/format.h>
#include "velox/common/encode/Base64.h"
#include "velox/expression/CastExpr.h"
#include "velox/functions/Udf.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/type/Conversions.h"
#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {
namespace detail {
static const StringView kNull = "NULL";

/// Style used to render binary values to strings. Mirrors Spark's
/// `spark.sql.binaryOutputStyle`.
enum class BinaryOutputStyle {
  // "[31 32]" -- hex digits separated by spaces and wrapped in square
  // brackets. Default when the Spark config is unset.
  kHexDiscrete,
  // "3132" -- uppercase hex digits with no separators.
  kHex,
  // "MTI" -- base64 encoded with no padding.
  kBase64,
  // "12" -- raw bytes interpreted as a UTF-8 string.
  kUtf8,
  // "[49, 50]" -- decimal byte values separated by ", " and wrapped in square
  // brackets.
  kBasic,
};

inline BinaryOutputStyle parseBinaryOutputStyle(const std::string& style) {
  if (style.empty() || style == "HEX_DISCRETE") {
    return BinaryOutputStyle::kHexDiscrete;
  }
  if (style == "HEX") {
    return BinaryOutputStyle::kHex;
  }
  if (style == "BASE64") {
    return BinaryOutputStyle::kBase64;
  }
  if (style == "UTF-8") {
    return BinaryOutputStyle::kUtf8;
  }
  if (style == "BASIC") {
    return BinaryOutputStyle::kBasic;
  }
  VELOX_USER_FAIL(
      "Unsupported value for binary output style: '{}'. "
      "Expected one of: HEX_DISCRETE, HEX, BASE64, UTF-8, BASIC.",
      style);
}
} // namespace detail

/// to_pretty_string(x) -> varchar
/// Returns pretty string for int8, int16, int32, int64, bool, Date, Varchar. It
/// has one difference with casting value to string:
/// 1) It prints null input as "NULL" rather than producing null output.
template <typename TExec>
struct ToPrettyStringFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/) {
    inputType_ = inputTypes[0];
  }

  template <typename TInput>
  Status callNullable(out_type<Varchar>& result, const TInput* input) {
    if (input) {
      if constexpr (std::is_same_v<TInput, StringView>) {
        result.setNoCopy(*input);
        return Status::OK();
      }
      if constexpr (std::is_same_v<TInput, int32_t>) {
        if (inputType_->isDate()) {
          try {
            auto output = DATE()->toString(*input);
            result.append(output);
          } catch (const std::exception& e) {
            return Status::Invalid(e.what());
          }
          return Status::OK();
        }
      }
      const auto castResult =
          util::Converter<TypeKind::VARCHAR, void, util::SparkCastPolicy>::
              tryCast(*input);
      VELOX_DCHECK(!castResult.hasError());
      result.copy_from(castResult.value());
    } else {
      result.setNoCopy(detail::kNull);
    }
    return Status::OK();
  }

 private:
  TypePtr inputType_;
};

/// Returns pretty string for varbinary. It has several differences with
/// cast(varbinary as string):
/// 1) It prints null input as "NULL" rather than producing null output.
/// 2) The binary value is rendered according to the Spark session config
///    `spark.binary_output_style` (see SparkQueryConfig::kBinaryOutputStyle).
///    Supported values are HEX_DISCRETE (default), HEX, BASE64, UTF-8, and
///    BASIC. Examples for the bytes [0x31, 0x32, 0x33]:
///      HEX_DISCRETE: "[31 32 33]"
///      HEX:          "313233"
///      BASE64:       "MTIz"
///      UTF-8:        "123"
///      BASIC:        "[49, 50, 51]"
template <typename TExec>
struct ToPrettyStringVarbinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      A* /*a*/) {
    style_ = detail::parseBinaryOutputStyle(
        SparkQueryConfig{config}.binaryOutputStyle());
  }

  template <typename TInput>
  void callNullable(out_type<Varchar>& result, const TInput* input) {
    if (!input) {
      result.setNoCopy(detail::kNull);
      return;
    }
    switch (style_) {
      case detail::BinaryOutputStyle::kHexDiscrete:
        writeHexDiscrete(result, *input);
        return;
      case detail::BinaryOutputStyle::kHex:
        writeHex(result, *input);
        return;
      case detail::BinaryOutputStyle::kBase64:
        writeBase64(result, *input);
        return;
      case detail::BinaryOutputStyle::kUtf8:
        result.setNoCopy(*input);
        return;
      case detail::BinaryOutputStyle::kBasic:
        writeBasic(result, *input);
        return;
    }
  }

 private:
  template <typename TInput>
  static void writeHexDiscrete(out_type<Varchar>& result, const TInput& input) {
    // "[XX XX XX]" -- 2 chars per byte + (size - 1) spaces + 2 brackets.
    // For empty input the result is just "[]".
    if (input.size() == 0) {
      result.resize(2);
      result.data()[0] = '[';
      result.data()[1] = ']';
      return;
    }
    result.resize(1 + 3 * input.size());
    char* pos = result.data();
    *pos++ = '[';
    for (size_t i = 0; i < input.size(); ++i) {
      fmt::format_to(pos, "{:02X}", static_cast<int>(input.data()[i]));
      pos += 2;
      *pos++ = ' ';
    }
    *--pos = ']';
  }

  template <typename TInput>
  static void writeHex(out_type<Varchar>& result, const TInput& input) {
    // "XXXX..." -- 2 uppercase hex chars per byte, no separators.
    result.resize(2 * input.size());
    char* pos = result.data();
    for (size_t i = 0; i < input.size(); ++i) {
      fmt::format_to(pos, "{:02X}", static_cast<int>(input.data()[i]));
      pos += 2;
    }
  }

  template <typename TInput>
  static void writeBase64(out_type<Varchar>& result, const TInput& input) {
    // Spark uses Base64 encoding without padding. Velox's Base64::encode
    // always includes padding, so strip the trailing '=' characters.
    std::string encoded = encoding::Base64::encode(input.data(), input.size());
    while (!encoded.empty() && encoded.back() == '=') {
      encoded.pop_back();
    }
    result.resize(encoded.size());
    if (!encoded.empty()) {
      std::memcpy(result.data(), encoded.data(), encoded.size());
    }
  }

  template <typename TInput>
  static void writeBasic(out_type<Varchar>& result, const TInput& input) {
    // "[d, d, d]" -- signed decimal byte values separated by ", ".
    if (input.size() == 0) {
      result.resize(2);
      result.data()[0] = '[';
      result.data()[1] = ']';
      return;
    }
    std::string buffer;
    buffer.reserve(2 + 5 * input.size());
    buffer.push_back('[');
    for (size_t i = 0; i < input.size(); ++i) {
      if (i > 0) {
        buffer.append(", ");
      }
      // Spark interprets bytes as signed `Byte` (-128..127), so 0xFF
      // renders as "-1". Cast through int8_t to force signed
      // interpretation regardless of whether plain `char` is signed on
      // the target platform.
      fmt::format_to(
          std::back_inserter(buffer),
          "{}",
          static_cast<int>(static_cast<int8_t>(input.data()[i])));
    }
    buffer.push_back(']');
    result.resize(buffer.size());
    std::memcpy(result.data(), buffer.data(), buffer.size());
  }

  detail::BinaryOutputStyle style_{detail::BinaryOutputStyle::kHexDiscrete};
};

/// Returns pretty string for Timestamp. It has one difference with
/// cast(timestamp as string):
/// 1) It prints null input as "NULL" rather than producing null output.
template <typename TExec>
struct ToPrettyStringTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Timestamp>* /*timestamp*/) {
    auto timezone = config.sessionTimezone();
    if (!timezone.empty()) {
      options_.timeZone = tz::locateZone(timezone);
    }
    timestampRowSize_ = getMaxStringLength(options_);
  }

  Status callNullable(
      out_type<Varchar>& result,
      const arg_type<Timestamp>* timestamp) {
    if (timestamp) {
      Timestamp inputValue(*timestamp);
      try {
        if (options_.timeZone) {
          inputValue.toTimezone(*(options_.timeZone));
        }
        result.reserve(timestampRowSize_);
        const auto stringView =
            Timestamp::tsToStringView(inputValue, options_, result.data());
        result.resize(stringView.size());
      } catch (const std::exception& e) {
        return Status::Invalid(
            "Invalid timestamp in to_pretty_string: {}", e.what());
      }
    } else {
      result.setNoCopy(detail::kNull);
    }
    return Status::OK();
  }

 private:
  TimestampToStringOptions options_ = {
      .precision = TimestampToStringOptions::Precision::kMicroseconds,
      .leadingPositiveSign = true,
      .skipTrailingZeros = true,
      .zeroPaddingYear = true,
      .dateTimeSeparator = ' ',
  };
  std::string::size_type timestampRowSize_;
};

/// Returns pretty string for short decimal and long decimal. It has one
/// difference with cast(decimal as string):
/// 1) It prints null input as "NULL" rather than producing null output.
template <typename TExec>
struct ToPrettyStringDecimalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/) {
    auto [precision, scale] = getDecimalPrecisionScale(*inputTypes[0]);
    precision_ = precision;
    scale_ = scale;
    maxRowSize_ = velox::DecimalUtil::maxStringViewSize(precision, scale);
  }

  template <typename TInput>
  void callNullable(out_type<Varchar>& result, const TInput* input) {
    if (input) {
      if (StringView::isInline(maxRowSize_)) {
        DecimalUtil::castToString<TInput>(
            *input, scale_, maxRowSize_, inlined_);
        result.setNoCopy(inlined_);
      } else {
        result.reserve(maxRowSize_);
        auto actualSize = DecimalUtil::castToString<TInput>(
            *input, scale_, maxRowSize_, result.data());
        result.resize(actualSize);
      }
    } else {
      result.setNoCopy(detail::kNull);
    }
  }

 private:
  uint8_t precision_;
  uint8_t scale_;
  int32_t maxRowSize_;
  char inlined_[StringView::kInlineSize];
};
} // namespace facebook::velox::functions::sparksql
