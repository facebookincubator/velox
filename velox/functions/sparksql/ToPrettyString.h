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
#include "velox/expression/CastExpr.h"
#include "velox/functions/Udf.h"
#include "velox/type/Conversions.h"
#include "velox/type/Timestamp.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {
namespace detail {
static const StringView kNull = "NULL";
}

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
/// 2) It prints binary value as hex string representation rather than UTF-8.
/// The pretty string is composed of the hex digits of bytes and spaces between
/// them. E.g., the result of to_pretty_string("abc") is "[31 32 33]".
template <typename TExec>
struct ToPrettyStringVarbinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename TInput>
  void callNullable(out_type<Varchar>& result, const TInput* input) {
    if (input) {
      // One byte spares 2 char, and with the spaces and the boxes.
      // Byte size: 2 * input->size(), spaces size: input->size() - 1, boxes
      // size: 2, its sum is 1 + 3 * input->size().
      result.resize(1 + 3 * input->size());
      char* const startPosition = result.data();
      char* pos = startPosition;
      *pos++ = '[';
      for (auto i = 0; i < input->size(); i++) {
        auto end =
            fmt::format_to(pos, "{:02X}", static_cast<int>(input->data()[i]));
        int count = end - pos;
        if (count != 2) {
          // Malformed input
          VELOX_USER_FAIL(
              "to_pretty_string(VARBINARY): failed to format byte at index {} (value: {}). Expected 2 chars, got {}.",
              i,
              static_cast<int>(input->data()[i]),
              count);
        }

        pos += 2;
        *pos++ = ' ';
      }
      *--pos = ']';
    } else {
      result.setNoCopy(detail::kNull);
    }
  }
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
        VELOX_USER_FAIL("Invalid timestamp in to_pretty_string: {}", e.what());
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
