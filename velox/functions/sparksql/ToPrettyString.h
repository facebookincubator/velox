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

#include "velox/expression/CastExpr-inl.h"
#include "velox/functions/Udf.h"
#include "velox/type/Conversions.h"

namespace facebook::velox::functions::sparksql {
namespace detail {
static const StringView kNull = "NULL";
}

/// toprettystring(x) -> varchar
/// Returns pretty string for int8, int16, int32, int64, bool, Date, Varchar. It
/// has one difference with casting value to string:
/// - It prints null values (either from column or struct field) as "NULL".
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
  void callNullable(out_type<Varchar>& result, const TInput* input) {
    if (input) {
      if constexpr (std::is_same_v<TInput, StringView>) {
        result.setNoCopy(*input);
        return;
      }
      if constexpr (std::is_same_v<TInput, int32_t>) {
        if (inputType_->isDate()) {
          auto output = DATE()->toString(*input);
          result.append(output);
          return;
        }
      }
      const auto castResult =
          util::Converter<TypeKind::VARCHAR, void, util::SparkCastPolicy>::
              tryCast(*input);
      if (castResult.hasError()) {
        result.setNoCopy(detail::kNull);
      } else {
        result.copy_from(castResult.value());
      }
    } else {
      result.setNoCopy(detail::kNull);
    }
  }

 private:
  TypePtr inputType_;
};

/// Returns pretty string for Varbinary. It has several differences with casting
/// value to string:
/// - It prints null values (either from column or struct field) as "NULL".
/// - It prints binary values (either from column or struct field) using the hex
/// format. Returns a pretty string of the byte array which prints each byte as
/// a hex digit and add spaces between them. For example, [1A C0].
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
        auto formated = fmt::format("{:X}", input->data()[i]);
        *pos++ = formated.data()[0];
        *pos++ = formated.data()[1];
        *pos++ = ' ';
      }
      *--pos = ']';
    } else {
      result.setNoCopy(detail::kNull);
    }
  }
};

/// Returns pretty string for Timestamp. It has one difference with casting
/// value to string:
/// - It prints null values (either from column or struct field) as "NULL".
template <typename TExec>
struct ToPrettyStringTimeStampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Timestamp>* /*timestamp*/) {
    auto timezone = config.sessionTimezone();
    if (!timezone.empty()) {
      options_.timeZone = date::locate_zone(timezone);
    }
    timestampRowSize_ = getMaxStringLength(options_);
  }

  void callNullable(
      out_type<Varchar>& result,
      const arg_type<Timestamp>* timestamp) {
    if (timestamp) {
      Timestamp inputValue(*timestamp);
      if (options_.timeZone) {
        inputValue.toTimezone(*(options_.timeZone));
      }
      result.reserve(timestampRowSize_);
      const auto stringView =
          Timestamp::tsToStringView(inputValue, options_, result.data());
      result.resize(stringView.size());
    } else {
      result.setNoCopy(detail::kNull);
    }
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
/// difference with casting value to string:
/// - It prints null values (either from column or struct field) as "NULL".
template <typename TExec>
struct ToPrettyStringDecimalFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/) {
    const auto [precision, scale] = getDecimalPrecisionScale(*inputTypes[0]);
    precision_ = precision;
    scale_ = scale;
    
    maxRowSize_ = velox::DecimalUtil::getMaxStringLength();
    if (scale > 0) {
      ++maxRowSize_; // A dot.
    }
    if (precision == scale) {
      ++maxRowSize_; // Leading zero.
    }
  }

  template <typename TInput>
  void callNullable(out_type<Varchar>& result, const TInput* input) {
    if (input) {
      if (StringView::isInline(maxRowSize_)) {
        auto view = exec::detail::convertToStringView<TInput>(
            *input, scale_, maxRowSize_, inlined_);
        result.setNoCopy(inlined_);
      } else {
        result.reserve(maxRowSize_);
        auto view = exec::detail::convertToStringView<TInput>(
            *input, scale_, maxRowSize_, result.data());
        result.resize(view.size());
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
