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

#include "velox/experimental/cudf/expression/CommonFunctions.h"
#include "velox/experimental/cudf/expression/DateTruncFunction.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/expression/sparksql/DateAddFunction.h"
#include "velox/experimental/cudf/expression/sparksql/HashFunction.h"
#include "velox/experimental/cudf/expression/sparksql/SubStringFunction.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/TimeUtils.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"

#include <cudf/strings/convert/convert_datetime.hpp>

#include <cctype>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

constexpr std::string_view kDateFormatName{"date_format"};

bool isDateFormatCall(std::string_view functionName) {
  return functionName.size() >= kDateFormatName.size() &&
      functionName.compare(
          functionName.size() - kDateFormatName.size(),
          kDateFormatName.size(),
          kDateFormatName) == 0;
}

bool containsDateFormatCall(const core::TypedExprPtr& expression) {
  const auto call =
      std::dynamic_pointer_cast<const core::CallTypedExpr>(expression);
  if (call && isDateFormatCall(call->name())) {
    return true;
  }
  for (const auto& input : expression->inputs()) {
    if (containsDateFormatCall(input)) {
      return true;
    }
  }
  return false;
}

void registerSparkArrayAccessFunctions(const std::string& prefix) {
  // Spark get is 0 based and returns NULL for negative or out-of-bounds
  // indices.
  registerArrayAccessFunction(
      prefix + "get",
      ArrayAccessPolicy{
          .allowNegativeIndices = true,
          .nullOnNegativeIndices = true,
          .allowOutOfBound = true,
          .indexStartsAtOne = false,
      },
      arrayAccessSignatures({"tinyint", "smallint", "integer", "bigint"}));
}

// Appends a literal character, escaping '%' as "%%" for cuDF's strftime
// parser.
void appendCudfLiteralCharacter(std::string& result, char character) {
  if (character == '%') {
    result += "%%";
  } else {
    result += character;
  }
}

// Maps one repeated Spark pattern token to its cuDF strftime directive. For
// example, "yyyy" maps to "%Y" and "MM" maps to "%m".
std::optional<std::string_view> cudfDateFormatToken(
    char token,
    size_t tokenLength) {
  if (token == 'y') {
    if (tokenLength == 4) {
      return "%Y";
    }
    if (tokenLength == 2) {
      return "%y";
    }
    return std::nullopt;
  }

  if (tokenLength != 2) {
    return std::nullopt;
  }

  switch (token) {
    case 'M':
      return "%m";
    case 'd':
      return "%d";
    case 'H':
      return "%H";
    case 'm':
      return "%M";
    case 's':
      return "%S";
    default:
      return std::nullopt;
  }
}

// Converts a complete Spark datetime pattern, including quoted literals, to
// cuDF strftime syntax. For example, "yyyy-MM-dd" becomes "%Y-%m-%d" and
// "yyyy'Q'MM" becomes "%YQ%m". Returns nullopt for valid Spark patterns that
// cuDF does not support and throws for malformed patterns.
std::optional<std::string> sparkToCudfDateFormat(std::string_view sparkFormat) {
  VELOX_USER_CHECK(!sparkFormat.empty(), "Invalid pattern specification");

  std::string result;
  result.reserve(sparkFormat.size() * 2);

  for (size_t i = 0; i < sparkFormat.size();) {
    const auto character = sparkFormat[i];
    if (character == '\'') {
      if (i + 1 < sparkFormat.size() && sparkFormat[i + 1] == '\'') {
        appendCudfLiteralCharacter(result, '\'');
        i += 2;
        continue;
      }

      ++i;
      bool closedLiteral{false};
      while (i < sparkFormat.size()) {
        if (sparkFormat[i] != '\'') {
          appendCudfLiteralCharacter(result, sparkFormat[i]);
          ++i;
          continue;
        }
        if (i + 1 < sparkFormat.size() && sparkFormat[i + 1] == '\'') {
          appendCudfLiteralCharacter(result, '\'');
          i += 2;
          continue;
        }
        ++i;
        closedLiteral = true;
        break;
      }
      if (!closedLiteral) {
        VELOX_USER_FAIL("No closing single quote for literal");
      }
      continue;
    }

    if (!std::isalpha(static_cast<unsigned char>(character))) {
      appendCudfLiteralCharacter(result, character);
      ++i;
      continue;
    }

    const auto tokenStart = i;
    while (i < sparkFormat.size() && sparkFormat[i] == character) {
      ++i;
    }
    const auto cudfToken = cudfDateFormatToken(character, i - tokenStart);
    if (!cudfToken.has_value()) {
      return std::nullopt;
    }
    result += cudfToken.value();
  }

  return result;
}

// Extracts and translates a non-null constant format argument. For example,
// date_format(c0, 'yyyy') produces "%Y".
std::optional<std::string> getCudfSparkDateFormat(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using velox::exec::ConstantExpr;

  if (expr->inputs().size() != 2) {
    return std::nullopt;
  }
  const auto formatExpr =
      std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
  if (!formatExpr || formatExpr->value()->isNullAt(0)) {
    return std::nullopt;
  }
  return sparkToCudfDateFormat(formatExpr->value()->toString(0));
}

bool canEvaluateDateFormat(
    const std::shared_ptr<velox::exec::Expr>& expression) {
  return getCudfSparkDateFormat(expression).has_value();
}

// Formats timestamps using the subset of Spark patterns supported by cuDF.
class DateFormatFunction : public CudfFunction {
 public:
  explicit DateFormatFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "date_format expects exactly 2 inputs");

    auto cudfFormat = getCudfSparkDateFormat(expr);
    VELOX_CHECK(
        cudfFormat.has_value(),
        "date_format format string must be a supported non-null constant");
    cudfFormat_ = std::move(cudfFormat.value());
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    const auto inputColumn = asView(inputColumns[0]);
    return cudf::strings::from_timestamps(
        inputColumn, cudfFormat_, cudf::strings_column_view{}, stream, mr);
  }

 private:
  // Uses cuDF strftime syntax after validation and translation at construction.
  std::string cudfFormat_;
};

} // namespace

bool containsSparkDateFormat(
    const std::vector<core::TypedExprPtr>& expressions) {
  for (const auto& expression : expressions) {
    if (containsDateFormatCall(expression)) {
      return true;
    }
  }
  return false;
}

bool requiresCpuSparkDateFormat(
    const std::vector<core::TypedExprPtr>& expressions,
    const core::QueryConfig& queryConfig) {
  const bool usesLegacyFormatter =
      functions::sparksql::SparkQueryConfig{queryConfig}.legacyDateFormatter();
  const auto* sessionTimeZone =
      facebook::velox::functions::getTimeZoneFromConfig(queryConfig);
  if (!usesLegacyFormatter &&
      (sessionTimeZone == nullptr || sessionTimeZone->id() == 0)) {
    return false;
  }
  return containsSparkDateFormat(expressions);
}

void registerSparkFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  const std::vector<exec::FunctionSignaturePtr> subStringSignatures{
      FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .argumentType("integer")
          .argumentType("integer")
          .build()};
  for (const auto& name : {prefix + "substr", prefix + "substring"}) {
    // Route both spellings to the Spark implementation in cuDF. Presto
    // substring is registered only when Presto functions are registered, so
    // Spark runtimes do not need to override an existing candidate.
    registerCudfFunction(
        name,
        [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
          return sparksql::makeSubStringFunction(expr);
        },
        subStringSignatures);
  }

  registerCudfFunction(
      prefix + "hash_with_seed",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<sparksql::HashFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .constantArgumentType("integer")
           .argumentType("any")
           .variableArity("any")
           .build()},
      true,
      sparksql::HashFunction::canEvaluate);

  registerCudfFunction(
      prefix + "date_add",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<sparksql::DateAddFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("integer")
           .build()});

  registerCudfFunction(
      prefix + "date_trunc",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<DateTruncFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp")
           .constantArgumentType("varchar")
           .argumentType("timestamp")
           .build()},
      true,
      DateTruncFunction::canEvaluate);

  registerCudfFunction(
      prefix + "date_format",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<DateFormatFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("timestamp")
           .constantArgumentType("varchar")
           .build()},
      /*overwrite=*/true,
      /*canEvaluate=*/canEvaluateDateFormat);

  registerSparkArrayAccessFunctions(prefix);
}

} // namespace facebook::velox::cudf_velox
