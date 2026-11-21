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

#include "velox/functions/sparksql/ToCsv.h"

#include <string_view>
#include <type_traits>
#include <utility>

#include <double-conversion/double-conversion.h>
#include <folly/String.h>

#include "velox/expression/DecodedArgs.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Controls CSV field separators, quoting, escaping, and null rendering.
struct CsvWriteOptions {
  char delimiter{','};
  char quote{'"'};
  char escape{'\\'};
  std::string_view nullValue;
};

enum class CsvOptionKind {
  kDynamic,
  kIgnored,
  kDelimiter,
  kQuote,
  kEscape,
  kNullValue,
  kError,
};

struct ParsedCsvOption {
  CsvOptionKind kind{CsvOptionKind::kIgnored};
  char character{0};
  std::string_view value;
};

struct PreparedCsvOption {
  CsvOptionKind kind;
  size_t argumentIndex{0};
  char character{0};
  std::string value;
};

Status parseCsvConfigArg(std::string_view argument, ParsedCsvOption& option) {
  const auto equalsPosition = argument.find('=');
  if (equalsPosition == std::string_view::npos) {
    return Status::UserError(
        "to_csv: option must use key=value format: {}", argument);
  }

  std::string key{argument.substr(0, equalsPosition)};
  folly::toLowerAscii(key);
  const auto value = argument.substr(equalsPosition + 1);
  if (key == "sep" || key == "delimiter") {
    if (value.size() != 1) {
      return Status::UserError(
          "to_csv: separator must be exactly 1 character, got '{}'", value);
    }
    option = {CsvOptionKind::kDelimiter, value[0], {}};
  } else if (key == "quote") {
    if (value.size() != 1) {
      return Status::UserError(
          "to_csv: quote must be exactly 1 character, got '{}'", value);
    }
    option = {CsvOptionKind::kQuote, value[0], {}};
  } else if (key == "escape") {
    if (value.size() != 1) {
      return Status::UserError(
          "to_csv: escape must be exactly 1 character, got '{}'", value);
    }
    option = {CsvOptionKind::kEscape, value[0], {}};
  } else if (key == "nullvalue") {
    option = {CsvOptionKind::kNullValue, 0, value};
  } else {
    return Status::UserError("to_csv: unsupported option: {}", key);
  }
  return Status::OK();
}

void applyCsvConfigArg(
    const ParsedCsvOption& option,
    CsvWriteOptions& options) {
  switch (option.kind) {
    case CsvOptionKind::kDelimiter:
      options.delimiter = option.character;
      return;
    case CsvOptionKind::kQuote:
      options.quote = option.character;
      return;
    case CsvOptionKind::kEscape:
      options.escape = option.character;
      return;
    case CsvOptionKind::kNullValue:
      options.nullValue = option.value;
      return;
    default:
      VELOX_UNREACHABLE();
  }
}

void applyCsvConfigArg(
    const PreparedCsvOption& option,
    CsvWriteOptions& options) {
  ParsedCsvOption parsedOption{
      option.kind, option.character, std::string_view{option.value}};
  applyCsvConfigArg(parsedOption, options);
}

void appendCsvString(
    std::string_view value,
    std::string& result,
    const CsvWriteOptions& options) {
  bool needsQuoting{false};
  for (char character : value) {
    if (character == options.delimiter || character == options.quote ||
        character == '\n' || character == '\r') {
      needsQuoting = true;
      break;
    }
  }

  size_t start{0};
  size_t end{value.size()};
  while (start < end && static_cast<unsigned char>(value[start]) <= ' ') {
    ++start;
  }
  while (end > start && static_cast<unsigned char>(value[end - 1]) <= ' ') {
    --end;
  }
  const auto trimmed = value.substr(start, end - start);

  if (trimmed.empty()) {
    result.append("\"\"");
    return;
  }

  if (!needsQuoting) {
    result.append(trimmed.data(), trimmed.size());
    return;
  }

  result += options.quote;
  for (char character : trimmed) {
    if (character == options.quote) {
      result += options.escape;
    } else if (options.escape != options.quote && character == options.escape) {
      result += options.escape;
    }
    result += character;
  }
  result += options.quote;
}

const double_conversion::DoubleToStringConverter& javaNumberConverter() {
  static const double_conversion::DoubleToStringConverter converter{
      double_conversion::DoubleToStringConverter::EMIT_TRAILING_DECIMAL_POINT |
          double_conversion::DoubleToStringConverter::
              EMIT_TRAILING_ZERO_AFTER_POINT,
      "Infinity",
      "NaN",
      'E',
      -3,
      7,
      0,
      0};
  return converter;
}

template <typename T>
void appendFloatingPoint(T value, std::string& result) {
  char buffer[32];
  double_conversion::StringBuilder builder{buffer, sizeof(buffer)};
  bool converted;
  if constexpr (std::is_same_v<T, float>) {
    converted = javaNumberConverter().ToShortestSingle(value, &builder);
  } else {
    converted = javaNumberConverter().ToShortest(value, &builder);
  }
  VELOX_CHECK(converted, "Failed to format a floating-point value.");
  const auto length = builder.position();
  builder.Finalize();
  const std::string_view formatted{buffer, static_cast<size_t>(length)};
  const auto exponentPosition = formatted.find('E');
  if (exponentPosition != std::string_view::npos &&
      formatted.find('.') == std::string_view::npos) {
    result.append(formatted.substr(0, exponentPosition));
    result.append(".0");
    result.append(formatted.substr(exponentPosition));
  } else {
    result.append(formatted);
  }
}

template <typename T>
T valueAt(const VectorPtr& vector, vector_size_t row) {
  return vector->as<SimpleVector<T>>()->valueAt(row);
}

void appendField(
    const VectorPtr& field,
    vector_size_t row,
    TypeKind kind,
    const CsvWriteOptions& options,
    std::string& result) {
  if (field->isNullAt(row)) {
    result.append(options.nullValue);
    return;
  }

  switch (kind) {
    case TypeKind::BOOLEAN:
      result.append(valueAt<bool>(field, row) ? "true" : "false");
      return;
    case TypeKind::TINYINT:
      result.append(std::to_string(valueAt<int8_t>(field, row)));
      return;
    case TypeKind::SMALLINT:
      result.append(std::to_string(valueAt<int16_t>(field, row)));
      return;
    case TypeKind::INTEGER:
      result.append(std::to_string(valueAt<int32_t>(field, row)));
      return;
    case TypeKind::BIGINT:
      result.append(std::to_string(valueAt<int64_t>(field, row)));
      return;
    case TypeKind::REAL:
      appendFloatingPoint(valueAt<float>(field, row), result);
      return;
    case TypeKind::DOUBLE:
      appendFloatingPoint(valueAt<double>(field, row), result);
      return;
    case TypeKind::VARCHAR: {
      const auto value = valueAt<StringView>(field, row);
      appendCsvString(
          std::string_view{value.data(), value.size()}, result, options);
      return;
    }
    default:
      VELOX_UNREACHABLE();
  }
}

void validateInputType(const TypePtr& inputType) {
  VELOX_USER_CHECK_EQ(
      inputType->kind(),
      TypeKind::ROW,
      "to_csv: input must be a ROW, got {}",
      inputType->toString());
  for (auto index = 0; index < inputType->size(); ++index) {
    const auto& childType = inputType->childAt(index);
    const bool supported = childType->equivalent(*BOOLEAN()) ||
        childType->equivalent(*TINYINT()) ||
        childType->equivalent(*SMALLINT()) ||
        childType->equivalent(*INTEGER()) || childType->equivalent(*BIGINT()) ||
        childType->equivalent(*REAL()) || childType->equivalent(*DOUBLE()) ||
        childType->equivalent(*VARCHAR());
    VELOX_USER_CHECK(
        supported,
        "to_csv: unsupported field type at index {}: {}",
        index,
        childType->toString());
  }
}

// Serializes a ROW value using Spark's default CSV writer semantics.
class ToCsvFunction final : public exec::VectorFunction {
 public:
  ToCsvFunction(
      TypePtr inputType,
      std::vector<PreparedCsvOption> preparedOptions)
      : inputType_{std::move(inputType)},
        preparedOptions_{std::move(preparedOptions)} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, VARCHAR(), result);
    auto* flatResult = result->asFlatVector<StringView>();

    exec::LocalDecodedVector decodedRow{context, *args[0], rows};
    auto* baseRow = decodedRow->base()->as<RowVector>();

    std::vector<exec::LocalDecodedVector> decodedOptions;
    decodedOptions.reserve(preparedOptions_.size());
    for (const auto& option : preparedOptions_) {
      if (option.kind == CsvOptionKind::kDynamic) {
        decodedOptions.emplace_back(context, *args[option.argumentIndex], rows);
      }
    }

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      if (decodedRow->isNullAt(row)) {
        result->setNull(row, true);
        return;
      }

      CsvWriteOptions options;
      size_t dynamicOptionIndex{0};
      for (const auto& preparedOption : preparedOptions_) {
        if (preparedOption.kind == CsvOptionKind::kIgnored) {
          continue;
        }
        if (preparedOption.kind == CsvOptionKind::kError) {
          context.setStatus(row, Status::UserError("{}", preparedOption.value));
          result->setNull(row, true);
          return;
        }
        if (preparedOption.kind != CsvOptionKind::kDynamic) {
          applyCsvConfigArg(preparedOption, options);
          continue;
        }

        const auto& decodedOption = decodedOptions[dynamicOptionIndex++];
        if (decodedOption->isNullAt(row)) {
          continue;
        }
        const auto value = decodedOption->valueAt<StringView>(row);
        ParsedCsvOption parsedOption;
        auto status = parseCsvConfigArg(
            std::string_view{value.data(), value.size()}, parsedOption);
        if (!status.ok()) {
          context.setStatus(row, std::move(status));
          result->setNull(row, true);
          return;
        }
        applyCsvConfigArg(parsedOption, options);
      }

      const auto baseRowIndex = decodedRow->index(row);
      std::string output;
      output.reserve(64);
      for (auto index = 0; index < inputType_->size(); ++index) {
        if (index > 0) {
          output += options.delimiter;
        }
        appendField(
            baseRow->childAt(index),
            baseRowIndex,
            inputType_->childAt(index)->kind(),
            options,
            output);
      }
      flatResult->set(row, StringView{output});
    });
  }

 private:
  // Provides the logical child types used to dispatch field serialization.
  const TypePtr inputType_;

  // Preserves option ordering while avoiding repeated parsing of constants.
  const std::vector<PreparedCsvOption> preparedOptions_;
};

} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>> toCsvSignatures() {
  return {exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .returnType("varchar")
              .argumentType("T")
              .variableArity("varchar")
              .build()};
}

std::shared_ptr<exec::VectorFunction> makeToCsv(
    const std::string& /*functionName*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK(!inputArgs.empty(), "to_csv: input type is required");
  validateInputType(inputArgs[0].type);

  std::vector<PreparedCsvOption> preparedOptions;
  preparedOptions.reserve(inputArgs.size() - 1);
  for (size_t index = 1; index < inputArgs.size(); ++index) {
    const auto& constantValue = inputArgs[index].constantValue;
    if (!constantValue) {
      preparedOptions.push_back({CsvOptionKind::kDynamic, index, 0, {}});
      continue;
    }
    if (constantValue->isNullAt(0)) {
      preparedOptions.push_back({CsvOptionKind::kIgnored, index, 0, {}});
      continue;
    }

    const auto value =
        constantValue->as<ConstantVector<StringView>>()->valueAt(0);
    ParsedCsvOption parsedOption;
    const auto status = parseCsvConfigArg(
        std::string_view{value.data(), value.size()}, parsedOption);
    if (!status.ok()) {
      preparedOptions.push_back(
          {CsvOptionKind::kError, index, 0, std::string{status.message()}});
      continue;
    }
    preparedOptions.push_back(
        {parsedOption.kind,
         index,
         parsedOption.character,
         std::string{parsedOption.value}});
  }
  return std::make_shared<ToCsvFunction>(
      inputArgs[0].type, std::move(preparedOptions));
}

} // namespace facebook::velox::functions::sparksql
