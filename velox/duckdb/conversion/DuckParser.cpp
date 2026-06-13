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
#include "velox/duckdb/conversion/DuckParser.h"
#include "velox/common/base/Exceptions.h"
#include "velox/duckdb/conversion/DuckConversion.h"
#include "velox/parse/Expressions.h"
#include "velox/type/Variant.h"

#include <duckdb.hpp> // @manual
#include <duckdb/parser/expression/between_expression.hpp> // @manual
#include <duckdb/parser/expression/case_expression.hpp> // @manual
#include <duckdb/parser/expression/cast_expression.hpp> // @manual
#include <duckdb/parser/expression/columnref_expression.hpp> // @manual
#include <duckdb/parser/expression/comparison_expression.hpp> // @manual
#include <duckdb/parser/expression/conjunction_expression.hpp> // @manual
#include <duckdb/parser/expression/constant_expression.hpp> // @manual
#include <duckdb/parser/expression/function_expression.hpp> // @manual
#include <duckdb/parser/expression/lambda_expression.hpp> // @manual
#include <duckdb/parser/expression/operator_expression.hpp> // @manual
#include <duckdb/parser/expression/window_expression.hpp> // @manual
#include <duckdb/parser/parser.hpp> // @manual
#include <duckdb/parser/parser_options.hpp> // @manual

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace facebook::velox::duckdb {

using ::duckdb::BetweenExpression;
using ::duckdb::CaseExpression;
using ::duckdb::CastExpression;
using ::duckdb::ColumnRefExpression;
using ::duckdb::ComparisonExpression;
using ::duckdb::ConjunctionExpression;
using ::duckdb::ConstantExpression;
using ::duckdb::ExpressionClass;
using ::duckdb::ExpressionType;
using ::duckdb::FunctionExpression;
using ::duckdb::interval_t;
using ::duckdb::LogicalType;
using ::duckdb::LogicalTypeId;
using ::duckdb::LogicalTypeIdToString;
using ::duckdb::OperatorExpression;
using ::duckdb::ParsedExpression;
using ::duckdb::Parser;
using ::duckdb::ParserOptions;
using ::duckdb::StringUtil;
using ::duckdb::Value;
using ::duckdb::WindowBoundary;
using ::duckdb::WindowExpression;

namespace {

core::ExprPtr parseExpr(ParsedExpression& expr, const ParseOptions& options);

std::string normalizeFuncName(std::string input) {
  static std::map<std::string, std::string> kLookup{
      {"+", "plus"},
      {"-", "minus"},
      {"*", "multiply"},
      {"/", "divide"},
      {"%", "mod"},
      {"<", "lt"},
      {"<=", "lte"},
      {">", "gt"},
      {">=", "gte"},
      {"=", "eq"},
      {"!", "not"},
      {"!=", "neq"},
      {"<>", "neq"},
      {"and", "and"},
      {"or", "or"},
      {"is", "is"},
      {"~~", "like"},
      {"!~~", "notlike"},
      {"like_escape", "like"},
      {"not_like_escape", "notlike"},
      {"IS DISTINCT FROM", "distinct_from"},
      {"count_star", "count"},
  };
  auto it = kLookup.find(input);
  return (it == kLookup.end()) ? input : it->second;
}

// Convert duckDB operator name to Velox function. Some expression types such as
// coalesce and subscript need special treatment because
// `ExpressionTypeToOperator` returns an empty string.
std::string duckOperatorToVelox(ExpressionType type) {
  switch (type) {
    case ExpressionType::OPERATOR_IS_NULL:
      return "is_null";
    case ExpressionType::OPERATOR_COALESCE:
      return "coalesce";
    case ExpressionType::ARRAY_EXTRACT:
      return "subscript";
    case ExpressionType::COMPARE_IN:
      return "in";
    case ExpressionType::OPERATOR_NOT:
      return "not";
    case ExpressionType::OPERATOR_TRY:
      return "try";
    default:
      return normalizeFuncName(ExpressionTypeToOperator(type));
  }
}

// SQL functions could be registered with different prefixes.
// This function returns a full Velox function name with the registered prefix.
std::string toFullFunctionName(
    const std::string& functionName,
    const std::string& prefix) {
  // Special forms are registered without a prefix.
  static const std::unordered_set<std::string> specialForms{
      "cast", "and", "coalesce", "if", "or", "switch", "try"};
  if (specialForms.count(functionName)) {
    return functionName;
  }
  return prefix + functionName;
}

std::optional<std::string> getAlias(const ParsedExpression& expr) {
  const auto& alias = expr.alias;
  return alias.empty() ? std::optional<std::string>() : alias;
}

std::shared_ptr<const core::CallExpr> callExpr(
    std::string name,
    std::vector<core::ExprPtr> params,
    std::optional<std::string> alias,
    const ParseOptions& options) {
  // DuckDB parser requires IF to have 3 arguments: condition, then-clause, and
  // else-clause. For example, `IF(a > b, 10)` doesn't parse correctly and must
  // be written as `IF(a > b, 10, null)`. Remove the redundant else-clause.
  if (name == "if") {
    if (params.back()->is(core::IExpr::Kind::kConstant) &&
        params.back()->as<core::ConstantExpr>()->type()->isUnknown()) {
      params.pop_back();
    }
  }

  return std::make_shared<const core::CallExpr>(
      toFullFunctionName(name, options.functionPrefix),
      std::move(params),
      std::move(alias));
}

std::shared_ptr<const core::CallExpr> callExpr(
    std::string name,
    const core::ExprPtr& param,
    std::optional<std::string> alias,
    const ParseOptions& options) {
  std::vector<core::ExprPtr> params = {param};
  return std::make_shared<const core::CallExpr>(
      toFullFunctionName(name, options.functionPrefix),
      std::move(params),
      std::move(alias));
}

std::shared_ptr<const core::ConstantExpr> intervalConstant(
    const interval_t& interval,
    std::optional<std::string> alias) {
  if (interval.months != 0 && interval.days == 0 && interval.micros == 0) {
    return std::make_shared<const core::ConstantExpr>(
        INTERVAL_YEAR_MONTH(), Variant(interval.months), alias);
  }
  if (interval.months != 0) {
    VELOX_NYI("Mixed year-month and day-time intervals are not supported");
  }
  return std::make_shared<const core::ConstantExpr>(
      INTERVAL_DAY_TIME(),
      Variant(interval.days * 24L * 60 * 60 * 1'000 + interval.micros / 1'000),
      alias);
}

// Parse a constant (1, 99.8, "string", etc).
core::ExprPtr parseConstantExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  auto& constantExpr = dynamic_cast<ConstantExpression&>(expr);
  auto& value = constantExpr.value;
  const auto alias = getAlias(expr);

  if (value.type().id() == LogicalTypeId::INTERVAL) {
    return intervalConstant(value.GetValue<interval_t>(), alias);
  }

  // This is a hack to make DuckDB more compatible with the old Koski-based
  // parser. By default literal integer constants in DuckDB parser are INTEGER,
  // while in Koski parser they were BIGINT.
  if (value.type().id() == LogicalTypeId::INTEGER &&
      options.parseIntegerAsBigint) {
    value = Value::BIGINT(value.GetValue<int32_t>());
  }

  if (options.parseDecimalAsDouble &&
      value.type().id() == duckdb::LogicalTypeId::DECIMAL) {
    value = Value::DOUBLE(value.GetValue<double>());
  }

  return std::make_shared<const core::ConstantExpr>(
      toVeloxType(value.type()), duckValueToVariant(value), alias);
}

// Parse a column reference (col1, "col2", tbl.col, etc).
core::ExprPtr parseColumnRefExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  const auto& colRefExpr = dynamic_cast<ColumnRefExpression&>(expr);
  if (!colRefExpr.IsQualified()) {
    return std::make_shared<const core::FieldAccessExpr>(
        colRefExpr.GetColumnName(), getAlias(expr));
  }
  return std::make_shared<const core::FieldAccessExpr>(
      colRefExpr.GetColumnName(),
      getAlias(expr),
      std::vector<core::ExprPtr>{std::make_shared<const core::FieldAccessExpr>(
          colRefExpr.GetTableName(), std::nullopt)});
}

namespace {

std::optional<double> extractNumeric(const Value& value) {
  if (value.IsNull()) {
    return std::nullopt;
  }

  switch (value.type().id()) {
    case LogicalTypeId::BIGINT:
      return value.GetValue<int64_t>();
    case LogicalTypeId::INTEGER:
      return value.GetValue<int32_t>();
    case LogicalTypeId::DECIMAL:
    case LogicalTypeId::DOUBLE:
      return value.DefaultCastAs(::duckdb::LogicalType(LogicalTypeId::DOUBLE))
          .GetValue<double>();
    default:
      return std::nullopt;
  }
}

std::optional<double> extractNumeric(ParsedExpression& expr) {
  if (auto* constInput = dynamic_cast<ConstantExpression*>(&expr)) {
    return extractNumeric(constInput->value);
  }
  if (auto* castInput = dynamic_cast<CastExpression*>(&expr)) {
    return extractNumeric(*castInput->child);
  }
  if (auto* functionInput = dynamic_cast<FunctionExpression*>(&expr)) {
    if (normalizeFuncName(functionInput->function_name) == "trunc" &&
        functionInput->children.size() == 1) {
      auto value = extractNumeric(*functionInput->children[0]);
      if (value.has_value()) {
        return std::trunc(value.value());
      }
    }
  }
  return std::nullopt;
}

std::optional<LogicalType> logicalTypeFromName(std::string name) {
  if (name.size() > 1 && name.front() == '"' && name.back() == '"') {
    name = name.substr(1, name.size() - 2);
  }
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return std::toupper(c);
  });

  if (name == "BOOLEAN" || name == "BOOL") {
    return LogicalType(LogicalTypeId::BOOLEAN);
  }
  if (name == "TINYINT") {
    return LogicalType(LogicalTypeId::TINYINT);
  }
  if (name == "SMALLINT") {
    return LogicalType(LogicalTypeId::SMALLINT);
  }
  if (name == "INTEGER" || name == "INT" || name == "SIGNED") {
    return LogicalType(LogicalTypeId::INTEGER);
  }
  if (name == "BIGINT") {
    return LogicalType(LogicalTypeId::BIGINT);
  }
  if (name == "REAL" || name == "FLOAT" || name == "FLOAT4") {
    return LogicalType(LogicalTypeId::FLOAT);
  }
  if (name == "DOUBLE" || name == "DOUBLE PRECISION" || name == "FLOAT8") {
    return LogicalType(LogicalTypeId::DOUBLE);
  }
  if (name == "VARCHAR" || name == "CHAR" || name == "BPCHAR" ||
      name == "TEXT" || name == "STRING") {
    return LogicalType(LogicalTypeId::VARCHAR);
  }
  if (name == "BLOB" || name == "BYTEA" || name == "VARBINARY") {
    return LogicalType(LogicalTypeId::BLOB);
  }
  if (name == "DATE") {
    return LogicalType(LogicalTypeId::DATE);
  }
  if (name == "TIME") {
    return LogicalType(LogicalTypeId::TIME);
  }
  if (name == "TIMESTAMP" || name == "DATETIME") {
    return LogicalType(LogicalTypeId::TIMESTAMP);
  }
  if (name == "INTERVAL") {
    return LogicalType(LogicalTypeId::INTERVAL);
  }
  return std::nullopt;
}

LogicalType resolveParsedType(LogicalType type) {
  if (auto resolvedType = logicalTypeFromName(type.ToString())) {
    return resolvedType.value();
  }
  return type;
}

std::optional<int64_t> extractInteger(const Value& value) {
  switch (value.type().id()) {
    case LogicalTypeId::TINYINT:
      return value.GetValue<int8_t>();
    case LogicalTypeId::SMALLINT:
      return value.GetValue<int16_t>();
    case LogicalTypeId::INTEGER:
      return value.GetValue<int32_t>();
    case LogicalTypeId::BIGINT:
      return value.GetValue<int64_t>();
    case LogicalTypeId::INTEGER_LITERAL:
      return value.GetValue<int64_t>();
    default:
      return std::nullopt;
  }
}

template <typename T>
T checkedIntegerCast(int64_t value, LogicalTypeId targetType) {
  VELOX_USER_CHECK(
      value >= static_cast<int64_t>(std::numeric_limits<T>::min()) &&
          value <= static_cast<int64_t>(std::numeric_limits<T>::max()),
      "Cannot cast IN-list constant {} to {}",
      value,
      LogicalTypeIdToString(targetType));
  return static_cast<T>(value);
}

std::optional<Value> castIntegerInListConstant(
    const Value& value,
    LogicalTypeId targetType) {
  const auto integer = extractInteger(value);
  if (!integer.has_value()) {
    return std::nullopt;
  }

  switch (targetType) {
    case LogicalTypeId::TINYINT:
      return Value::TINYINT(
          checkedIntegerCast<int8_t>(integer.value(), targetType));
    case LogicalTypeId::SMALLINT:
      return Value::SMALLINT(
          checkedIntegerCast<int16_t>(integer.value(), targetType));
    case LogicalTypeId::INTEGER:
      return Value::INTEGER(
          checkedIntegerCast<int32_t>(integer.value(), targetType));
    case LogicalTypeId::BIGINT:
      return Value::BIGINT(integer.value());
    default:
      return std::nullopt;
  }
}

Value castInListConstant(const Value& value, const CastExpression& castExpr) {
  const auto sourceType = value.type().id();
  const auto targetLogicalType = resolveParsedType(castExpr.cast_type);
  const auto targetType = targetLogicalType.id();

  if (auto integerValue = castIntegerInListConstant(value, targetType)) {
    return integerValue.value();
  }

  if (sourceType == LogicalTypeId::VARCHAR) {
    const auto str = value.GetValue<std::string>();
    if (targetType == LogicalTypeId::BOOLEAN) {
      if (str == "t" || str == "true") {
        return Value::BOOLEAN(true);
      }
      if (str == "f" || str == "false") {
        return Value::BOOLEAN(false);
      }
    }
    if (targetType == LogicalTypeId::DATE) {
      return Value::DATE(::duckdb::Date::FromString(str));
    }
    if (targetType == LogicalTypeId::BLOB) {
      return Value::BLOB_RAW(str);
    }
  }

  if (sourceType == LogicalTypeId::DECIMAL &&
      targetType == LogicalTypeId::FLOAT) {
    return Value::FLOAT(static_cast<float>(value.GetValue<double>()));
  }

  return value.DefaultCastAs(targetLogicalType, !castExpr.try_cast);
}
} // namespace

std::shared_ptr<const core::ConstantExpr> tryParseInterval(
    const std::string& functionName,
    ParsedExpression& input,
    std::optional<std::string> alias) {
  auto value = extractNumeric(input);
  if (!value.has_value()) {
    return nullptr;
  }

  int64_t multiplier;

  if (functionName == "to_days") {
    multiplier = 24 * 60 * 60 * 1'000;
  } else if (functionName == "to_hours") {
    multiplier = 60 * 60 * 1'000;
  } else if (functionName == "to_minutes") {
    multiplier = 60 * 1'000;
  } else if (functionName == "to_seconds") {
    multiplier = 1'000;
  } else if (functionName == "to_milliseconds") {
    multiplier = 1;
  }
  // The other two options are years and months. They are expressed in terms of
  // number of months, and return a different type (INTERVAL_YEAR_MONTH).
  else {
    if (functionName == "to_years") {
      multiplier = 12;
    } else if (functionName == "to_months") {
      multiplier = 1;
    } else {
      return nullptr;
    }
    return std::make_shared<core::ConstantExpr>(
        INTERVAL_YEAR_MONTH(),
        Variant((int32_t)(value.value() * multiplier)),
        alias);
  }
  const auto millis = static_cast<int64_t>(value.value() * multiplier);
  return std::make_shared<core::ConstantExpr>(
      INTERVAL_DAY_TIME(), Variant(millis), alias);
}

// DuckDB parses struct literals {'x': 1, 'y': 2} as struct_pack(1 AS x, 2 AS
// y) and ROW(1, 2) as row(1, 2). Folds into a ROW constant when all arguments
// are constants. Returns nullptr otherwise.
core::ExprPtr tryFoldRowConstant(
    const std::vector<core::ExprPtr>& inputs,
    const std::optional<std::string>& alias) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  std::vector<Variant> values;
  names.reserve(inputs.size());
  types.reserve(inputs.size());
  values.reserve(inputs.size());
  for (const auto& input : inputs) {
    auto* constant = input->as<core::ConstantExpr>();
    if (!constant) {
      return nullptr;
    }
    names.push_back(constant->alias().value_or(""));
    types.push_back(constant->type());
    values.push_back(constant->value());
  }
  return std::make_shared<const core::ConstantExpr>(
      ROW(std::move(names), std::move(types)),
      Variant::row(std::move(values)),
      alias);
}

// DuckDB parses [1, 2, 3] as list_value(1, 2, 3). Folds into an ARRAY constant
// when all arguments are constants. Returns nullptr otherwise.
core::ExprPtr tryFoldArrayConstant(
    const std::vector<core::ExprPtr>& inputs,
    const std::optional<std::string>& alias) {
  std::vector<Variant> elements;
  elements.reserve(inputs.size());
  TypePtr elementType = UNKNOWN();
  for (const auto& input : inputs) {
    auto* constant = input->as<core::ConstantExpr>();
    if (!constant) {
      return nullptr;
    }
    elements.push_back(constant->value());
    if (!constant->value().isNull()) {
      elementType = constant->type();
    }
  }
  return std::make_shared<const core::ConstantExpr>(
      ARRAY(elementType), Variant::array(std::move(elements)), alias);
}

// Parse a function call (avg(a), func(1, b), etc).
// Arithmetic operators also follow this path (a + b, a * b, etc).
core::ExprPtr parseFunctionExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  const auto& functionExpr = dynamic_cast<FunctionExpression&>(expr);
  auto func = normalizeFuncName(functionExpr.function_name);

  if (functionExpr.children.size() == 1) {
    if (auto interval =
            tryParseInterval(func, *functionExpr.children[0], getAlias(expr))) {
      return interval;
    }
  }

  std::vector<core::ExprPtr> params;
  params.reserve(functionExpr.children.size());

  for (const auto& c : functionExpr.children) {
    params.emplace_back(parseExpr(*c, options));
  }

  if (func == "struct_pack" || func == "row") {
    if (auto rowConstant = tryFoldRowConstant(params, getAlias(expr))) {
      return rowConstant;
    }
  }

  // DuckDB parses [1, 2, 3] as list_value(1, 2, 3). Fold into an ARRAY
  // constant when all arguments are constants.
  if (func == "list_value") {
    if (auto arrayConstant = tryFoldArrayConstant(params, getAlias(expr))) {
      return arrayConstant;
    }
  }

  // NOT LIKE function needs special handling as it maps to two functions
  // "not" and "like".
  if (func == "notlike") {
    auto likeParams = params;
    params.clear();
    params.emplace_back(callExpr("like", std::move(likeParams), {}, options));
    func = "not";
  }
  return callExpr(func, std::move(params), getAlias(expr), options);
}

// Parse a comparison (a > b, a = b, etc).
core::ExprPtr parseComparisonExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  const auto& compExpr = dynamic_cast<ComparisonExpression&>(expr);
  std::vector<core::ExprPtr> params{
      parseExpr(*compExpr.left, options), parseExpr(*compExpr.right, options)};
  return callExpr(
      normalizeFuncName(ExpressionTypeToOperator(expr.GetExpressionType())),
      std::move(params),
      getAlias(expr),
      options);
}

// Parse x between lower and upper
core::ExprPtr parseBetweenExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  const auto& betweenExpr = dynamic_cast<BetweenExpression&>(expr);
  return callExpr(
      "between",
      {parseExpr(*betweenExpr.input, options),
       parseExpr(*betweenExpr.lower, options),
       parseExpr(*betweenExpr.upper, options)},
      getAlias(expr),
      options);
}

// Parse a conjunction (AND or OR).
core::ExprPtr parseConjunctionExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  const auto& conjExpr = dynamic_cast<ConjunctionExpression&>(expr);
  std::string conjName =
      StringUtil::Lower(ExpressionTypeToOperator(expr.GetExpressionType()));

  if (conjExpr.children.size() < 2) {
    throw std::invalid_argument(
        folly::sformat(
            "Malformed conjunction expression "
            "(expected at least 2 input columns, got {}).",
            conjExpr.children.size()));
  }

  // DuckDB's parser returns conjunction involving multiple input in a flat
  // expression, in the form `AND(a, b, d, e)`, but internally we expect
  // conjunctions to have exactly 2 input. This code converts that input into
  // `AND(AND(AND(a, b), d), e)` (so it's executed in the same order).
  core::ExprPtr current;
  for (size_t i = 1; i < conjExpr.children.size(); ++i) {
    std::vector<core::ExprPtr> params;
    params.reserve(2);

    if (current == nullptr) {
      params.emplace_back(parseExpr(*conjExpr.children[0], options));
      params.emplace_back(parseExpr(*conjExpr.children[1], options));
    } else {
      params.emplace_back(current);
      params.emplace_back(parseExpr(*conjExpr.children[i], options));
    }
    current = callExpr(conjName, std::move(params), getAlias(expr), options);
  }
  return current;
}

static bool areAllChildrenConstant(const OperatorExpression& operExpr) {
  for (const auto& child : operExpr.children) {
    if (child->GetExpressionType() != ExpressionType::VALUE_CONSTANT) {
      return false;
    }
  }
  return true;
}

// Parse an "operator", like NOT.
core::ExprPtr parseOperatorExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  const auto& operExpr = dynamic_cast<OperatorExpression&>(expr);

  // Code for array literal parsing (e.g. "ARRAY[1, 2, 3]")
  if (expr.GetExpressionType() == ExpressionType::ARRAY_CONSTRUCTOR) {
    if (areAllChildrenConstant(operExpr)) {
      std::vector<Variant> arrayElements;
      arrayElements.reserve(operExpr.children.size());

      TypePtr valueType = UNKNOWN();
      for (const auto& child : operExpr.children) {
        if (auto constantExpr =
                dynamic_cast<ConstantExpression*>(child.get())) {
          auto& value = constantExpr->value;
          if (value.type().id() == LogicalTypeId::INTEGER &&
              options.parseIntegerAsBigint) {
            value = Value::BIGINT(value.GetValue<int32_t>());
          }
          if (options.parseDecimalAsDouble &&
              value.type().id() == duckdb::LogicalTypeId::DECIMAL) {
            value = Value::DOUBLE(value.GetValue<double>());
          }
          arrayElements.emplace_back(duckValueToVariant(value));
          if (!value.IsNull()) {
            valueType = toVeloxType(value.type());
          }
        } else {
          VELOX_UNREACHABLE();
        }
      }
      return std::make_shared<const core::ConstantExpr>(
          ARRAY(valueType), Variant::array(arrayElements), getAlias(expr));
    } else {
      std::vector<core::ExprPtr> params;
      params.reserve(operExpr.children.size());

      for (const auto& child : operExpr.children) {
        params.emplace_back(parseExpr(*child, options));
      }
      return callExpr(
          "array_constructor", std::move(params), getAlias(expr), options);
    }
  }

  // Check if the operator is "IN" or "NOT IN".
  if (expr.GetExpressionType() == ExpressionType::COMPARE_IN ||
      expr.GetExpressionType() == ExpressionType::COMPARE_NOT_IN) {
    auto numValues = operExpr.children.size() - 1;

    std::vector<Variant> values;
    if (options.parseInListAsArray) {
      values.reserve(numValues);
    }

    std::vector<core::ExprPtr> params;
    if (!options.parseInListAsArray) {
      params.reserve(numValues + 1);
    }
    params.emplace_back(parseExpr(*operExpr.children[0], options));

    TypePtr valueType = UNKNOWN();
    for (auto i = 0; i < numValues; i++) {
      auto valueExpr = operExpr.children[i + 1].get();
      if (const auto castExpr = dynamic_cast<CastExpression*>(valueExpr)) {
        if (castExpr->child->GetExpressionType() ==
            ExpressionType::VALUE_CONSTANT) {
          auto constExpr =
              dynamic_cast<ConstantExpression*>(castExpr->child.get());
          auto value = castInListConstant(constExpr->value, *castExpr);
          if (options.parseInListAsArray) {
            values.emplace_back(duckValueToVariant(value));
            valueType = toVeloxType(castExpr->cast_type);
          } else {
            params.emplace_back(parseExpr(*castExpr->child, options));
          }
          continue;
        }
      }

      if (auto constantExpr = dynamic_cast<ConstantExpression*>(valueExpr)) {
        if (options.parseInListAsArray) {
          auto& value = constantExpr->value;
          if (options.parseDecimalAsDouble &&
              value.type().id() == duckdb::LogicalTypeId::DECIMAL) {
            value = Value::DOUBLE(value.GetValue<double>());
          }
          values.emplace_back(duckValueToVariant(value));
          if (!value.IsNull()) {
            valueType = toVeloxType(value.type());
          }
        } else {
          params.emplace_back(parseExpr(*constantExpr, options));
        }
        continue;
      }

      VELOX_UNSUPPORTED("IN list values need to be constant");
    }

    if (options.parseInListAsArray) {
      params.emplace_back(
          std::make_shared<const core::ConstantExpr>(
              ARRAY(valueType), Variant::array(values), std::nullopt));
    }
    auto inExpr = callExpr("in", std::move(params), getAlias(expr), options);
    // Translate COMPARE_NOT_IN into NOT(IN()).
    return (expr.GetExpressionType() == ExpressionType::COMPARE_IN)
        ? inExpr
        : callExpr("not", inExpr, std::nullopt, options);
  }

  std::vector<core::ExprPtr> params;
  params.reserve(operExpr.children.size());

  for (const auto& child : operExpr.children) {
    params.emplace_back(parseExpr(*child, options));
  }

  // STRUCT_EXTRACT(struct, 'entry') resolves nested field access such as
  // (a).b.c, (a.b).c
  if (expr.GetExpressionType() == ExpressionType::STRUCT_EXTRACT) {
    VELOX_CHECK_EQ(params.size(), 2);
    std::vector<core::ExprPtr> input = {params[0]};

    if (auto constantExpr =
            std::dynamic_pointer_cast<const core::ConstantExpr>(params[1])) {
      auto fieldName = constantExpr->value().value<std::string>();

      return std::make_shared<const core::FieldAccessExpr>(
          fieldName, getAlias(expr), std::move(input));
    } else {
      VELOX_UNSUPPORTED("STRUCT_EXTRACT field name must be constant");
    }
  }

  if (expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL) {
    return callExpr(
        "not",
        callExpr("is_null", std::move(params), std::nullopt, options),
        getAlias(expr),
        options);
  }

  return callExpr(
      duckOperatorToVelox(expr.GetExpressionType()),
      std::move(params),
      getAlias(expr),
      options);
}

namespace {
bool isNullConstant(const core::ExprPtr& expr) {
  if (auto constExpr =
          std::dynamic_pointer_cast<const core::ConstantExpr>(expr)) {
    return constExpr->value().isNull();
  }

  return false;
}
} // namespace

// Parse an IF()/CASE expression.
core::ExprPtr parseCaseExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  const auto& caseExpr = dynamic_cast<CaseExpression&>(expr);
  const auto& checks = caseExpr.case_checks;

  if (checks.size() == 1) {
    const auto& check = checks.front();

    std::vector<core::ExprPtr> params{
        parseExpr(*check.when_expr, options),
        parseExpr(*check.then_expr, options),
        parseExpr(*caseExpr.else_expr, options),
    };
    return callExpr("if", std::move(params), getAlias(expr), options);
  }

  // Detect simple CASE: DuckDB rewrites `CASE subject WHEN val THEN res` into
  // a searched CASE where each when_expr is `COMPARE_EQUAL(subject_copy, val)`.
  // The original form is lost in the AST (DuckDB's CaseExpression has no field
  // to distinguish the two). We reverse-engineer it by checking whether all
  // when_exprs are equality comparisons with a structurally identical left-hand
  // side (the subject), and if so emit "case" instead of "switch". CaseExpr
  // evaluates the subject exactly once and reuses the cached vector for each
  // WHEN comparison, avoiding re-evaluation of non-deterministic subjects
  // (e.g. rand()).
  //
  // Note: this detection can also match a user-written searched CASE whose
  // conditions happen to be `x = v1`, `x = v2`, ... with the same LHS. This
  // is semantically equivalent to `CASE x WHEN v1 ... WHEN v2 ...` for
  // deterministic subjects. For non-deterministic subjects the evaluation
  // count changes from per-branch to once, but writing a searched CASE with
  // repeated non-deterministic equality checks against the same expression
  // is unlikely in practice.
  {
    bool allEqWithSameSubject = true;
    const ComparisonExpression* firstComp =
        dynamic_cast<const ComparisonExpression*>(checks[0].when_expr.get());
    const std::string subjectStr =
        (firstComp && firstComp->type == ExpressionType::COMPARE_EQUAL)
        ? firstComp->left->ToString()
        : "";

    if (!subjectStr.empty()) {
      for (size_t i = 1; i < checks.size(); i++) {
        const auto* comp = dynamic_cast<const ComparisonExpression*>(
            checks[i].when_expr.get());
        if (!comp || comp->type != ExpressionType::COMPARE_EQUAL ||
            comp->left->ToString() != subjectStr) {
          allEqWithSameSubject = false;
          break;
        }
      }
    } else {
      allEqWithSameSubject = false;
    }

    if (allEqWithSameSubject) {
      // Emit: case(subject, when_val_0, then_0, when_val_1, then_1,
      //                       ..., [else])
      std::vector<core::ExprPtr> inputs;
      inputs.reserve(checks.size() * 2 + 2);
      // Subject — parse from the left side of the first condition.
      inputs.emplace_back(parseExpr(*firstComp->left, options));
      for (const auto& check : checks) {
        const auto& comp =
            dynamic_cast<const ComparisonExpression&>(*check.when_expr);
        inputs.emplace_back(parseExpr(*comp.right, options)); // WHEN value
        inputs.emplace_back(parseExpr(*check.then_expr, options)); // THEN
      }
      auto elseExpr = parseExpr(*caseExpr.else_expr, options);
      if (!isNullConstant(elseExpr)) {
        inputs.emplace_back(elseExpr);
      }
      return callExpr("case", std::move(inputs), getAlias(expr), options);
    }
  }

  // Searched CASE (or simple CASE that could not be detected): emit "switch".
  std::vector<core::ExprPtr> inputs;
  inputs.reserve(checks.size() * 2 + 1);
  for (auto& check : checks) {
    inputs.emplace_back(parseExpr(*check.when_expr, options));
    inputs.emplace_back(parseExpr(*check.then_expr, options));
  }

  auto elseExpr = parseExpr(*caseExpr.else_expr, options);
  if (!isNullConstant(elseExpr)) {
    inputs.emplace_back(elseExpr);
  }

  return callExpr("switch", std::move(inputs), getAlias(expr), options);
}

// Parse an CAST expression.
core::ExprPtr parseCastExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  const auto& castExpr = dynamic_cast<CastExpression&>(expr);
  std::vector<core::ExprPtr> params{parseExpr(*castExpr.child, options)};
  // We may need to expand toVeloxType in the future to support
  // Map and Array and Struct properly.
  auto targetType = toVeloxType(castExpr.cast_type);
  VELOX_CHECK(!params.empty());

  // Convert cast(NULL as <type>) into a constant NULL.
  if (auto* constant =
          dynamic_cast<const core::ConstantExpr*>(params[0].get())) {
    if (constant->value().isNull()) {
      return std::make_shared<const core::ConstantExpr>(
          targetType, Variant::null(targetType->kind()), getAlias(expr));
    }

    if (castExpr.cast_type.id() == LogicalTypeId::INTERVAL &&
        constant->type()->isVarchar()) {
      auto value = Value(constant->value().value<TypeKind::VARCHAR>())
                       .DefaultCastAs(castExpr.cast_type, !castExpr.try_cast);
      if (value.IsNull()) {
        return std::make_shared<const core::ConstantExpr>(
            targetType, Variant::null(targetType->kind()), getAlias(expr));
      }
      return intervalConstant(value.GetValue<interval_t>(), getAlias(expr));
    }

    // DuckDB parses BOOLEAN literal as cast expression.  Try to restore it back
    // to constant expression here.
    if (targetType->isBoolean() && constant->type()->isVarchar()) {
      const auto& value = constant->value();
      const auto& s = value.value<TypeKind::VARCHAR>();

      if (s == "t") {
        return std::make_shared<const core::ConstantExpr>(
            BOOLEAN(),
            Variant::create<TypeKind::BOOLEAN>(true),
            getAlias(expr));
      }

      if (s == "f") {
        return std::make_shared<const core::ConstantExpr>(
            BOOLEAN(),
            Variant::create<TypeKind::BOOLEAN>(false),
            getAlias(expr));
      }
    }

    // DuckDB parses DATE '...' and '...'::date as cast(varchar as DATE).
    // Fold into a DATE constant.
    if (targetType->isDate() && constant->type()->isVarchar()) {
      const auto& value = constant->value().value<TypeKind::VARCHAR>();
      return std::make_shared<const core::ConstantExpr>(
          DATE(),
          Variant::create<TypeKind::INTEGER>(DATE()->toDays(value)),
          getAlias(expr));
    }

    if (targetType->isVarbinary() && constant->type()->isVarchar()) {
      return std::make_shared<const core::ConstantExpr>(
          VARBINARY(),
          Variant::binary(constant->value().value<TypeKind::VARCHAR>()),
          getAlias(expr));
    }

    // ROW(1, 2)::struct(x bigint, y bigint) — re-type the ROW constant with
    // the target type (which carries field names). Child types must match.
    if (targetType->isRow() && targetType->equivalent(*constant->type())) {
      return std::make_shared<const core::ConstantExpr>(
          targetType, constant->value(), getAlias(expr));
    }
  }

  const bool isTryCast = castExpr.try_cast;
  return std::make_shared<const core::CastExpr>(
      targetType, params[0], isTryCast, getAlias(expr));
}

core::ExprPtr parseLambdaExpr(
    ParsedExpression& expr,
    const ParseOptions& options) {
  const auto& lambdaExpr = dynamic_cast<::duckdb::LambdaExpression&>(expr);
  auto capture = parseExpr(*lambdaExpr.lhs, options);
  auto body = parseExpr(*lambdaExpr.expr, options);

  // capture is either a core::FieldAccessExpr or a 'row' core::CallExpr with 2
  // or more core::FieldAccessExpr inputs.

  std::vector<std::string> names;
  if (auto fieldExpr =
          std::dynamic_pointer_cast<const core::FieldAccessExpr>(capture)) {
    names.push_back(fieldExpr->name());
  } else if (
      auto callExpr =
          std::dynamic_pointer_cast<const core::CallExpr>(capture)) {
    VELOX_CHECK_EQ(
        toFullFunctionName("row", options.functionPrefix), callExpr->name());
    for (auto& input : callExpr->inputs()) {
      auto fieldExpr =
          std::dynamic_pointer_cast<const core::FieldAccessExpr>(input);
      VELOX_CHECK_NOT_NULL(fieldExpr);
      names.push_back(fieldExpr->name());
    }
  } else {
    VELOX_FAIL(
        "Unexpected left-hand-side expression for the lambda expression: {}",
        capture->toString());
  }

  return std::make_shared<const core::LambdaExpr>(
      std::move(names), std::move(body));
}

core::ExprPtr parseExpr(ParsedExpression& expr, const ParseOptions& options) {
  switch (expr.GetExpressionClass()) {
    case ExpressionClass::CONSTANT:
      return parseConstantExpr(expr, options);

    case ExpressionClass::COLUMN_REF:
      return parseColumnRefExpr(expr, options);

    case ExpressionClass::FUNCTION:
      return parseFunctionExpr(expr, options);

    case ExpressionClass::COMPARISON:
      return parseComparisonExpr(expr, options);

    case ExpressionClass::BETWEEN:
      return parseBetweenExpr(expr, options);

    case ExpressionClass::CONJUNCTION:
      return parseConjunctionExpr(expr, options);

    case ExpressionClass::OPERATOR:
      return parseOperatorExpr(expr, options);

    case ExpressionClass::CASE:
      return parseCaseExpr(expr, options);

    case ExpressionClass::CAST:
      return parseCastExpr(expr, options);

    case ExpressionClass::LAMBDA:
      return parseLambdaExpr(expr, options);

    default:
      throw std::invalid_argument(
          "Unsupported expression type for DuckDB -> velox conversion: " +
          ::duckdb::ExpressionTypeToString(expr.GetExpressionType()));
  }
}

::duckdb::vector<::duckdb::unique_ptr<::duckdb::ParsedExpression>>
parseExpression(const std::string& exprString) {
  ParserOptions options;
  options.preserve_identifier_case = false;

  try {
    return Parser::ParseExpressionList(exprString, options);
  } catch (const std::exception& e) {
    VELOX_FAIL("Cannot parse expression: {}. {}", exprString, e.what());
  }
}

std::unique_ptr<::duckdb::ParsedExpression> parseSingleExpression(
    const std::string& exprString) {
  auto parsed = parseExpression(exprString);
  VELOX_CHECK_EQ(
      1, parsed.size(), "Expected exactly one expression: {}.", exprString);
  auto result = std::move(parsed.front());
  return result;
}
} // namespace

core::ExprPtr parseExpr(
    const std::string& exprString,
    const ParseOptions& options) {
  auto parsed = parseSingleExpression(exprString);
  return parseExpr(*parsed, options);
}

std::vector<core::ExprPtr> parseMultipleExpressions(
    const std::string& exprString,
    const ParseOptions& options) {
  auto parsedExpressions = parseExpression(exprString);
  VELOX_CHECK_GT(parsedExpressions.size(), 0);
  std::vector<core::ExprPtr> exprs;
  exprs.reserve(parsedExpressions.size());
  for (const auto& parsedExpr : parsedExpressions) {
    exprs.push_back(parseExpr(*parsedExpr, options));
  }
  return exprs;
}

namespace {
bool isAscending(::duckdb::OrderType orderType, const std::string& exprString) {
  switch (orderType) {
    case ::duckdb::OrderType::ASCENDING:
      return true;
    case ::duckdb::OrderType::DESCENDING:
      return false;
    case ::duckdb::OrderType::ORDER_DEFAULT:
      // ASC is the default.
      return true;
    case ::duckdb::OrderType::INVALID:
    default:
      VELOX_FAIL("Cannot parse ORDER BY clause: {}", exprString);
  }
}

bool isNullsFirst(
    ::duckdb::OrderByNullType orderByNullType,
    const std::string& exprString) {
  switch (orderByNullType) {
    case ::duckdb::OrderByNullType::NULLS_FIRST:
      return true;
    case ::duckdb::OrderByNullType::NULLS_LAST:
      return false;
    case ::duckdb::OrderByNullType::ORDER_DEFAULT:
      // NULLS LAST is the default.
      return false;
    case ::duckdb::OrderByNullType::INVALID:
    default:
      VELOX_FAIL("Cannot parse ORDER BY clause: {}", exprString);
  }

  VELOX_UNREACHABLE();
}
} // namespace

parse::OrderByClause parseOrderByExpr(const std::string& exprString) {
  ParserOptions options;
  ParseOptions parseOptions;
  options.preserve_identifier_case = false;
  auto orderByNodes = Parser::ParseOrderList(exprString, options);
  VELOX_CHECK_EQ(
      1,
      orderByNodes.size(),
      "Expected exactly one expression: {}.",
      exprString);

  const auto& orderByNode = orderByNodes[0];

  const bool ascending = isAscending(orderByNode.type, exprString);
  const bool nullsFirst = isNullsFirst(orderByNode.null_order, exprString);

  return {
      .expr = parseExpr(*orderByNode.expression, parseOptions),
      .ascending = ascending,
      .nullsFirst = nullsFirst};
}

core::AggregateCallExprPtr parseAggregateExpr(
    const std::string& exprString,
    const ParseOptions& options) {
  auto parsedExpr = parseSingleExpression(exprString);

  auto& functionExpr = dynamic_cast<FunctionExpression&>(*parsedExpr);

  auto callExpr = parseExpr(*parsedExpr, options);

  std::vector<core::SortKey> orderBy;
  if (functionExpr.order_bys) {
    for (const auto& orderByNode : functionExpr.order_bys->orders) {
      orderBy.push_back(
          {parseExpr(*orderByNode.expression, options),
           isAscending(orderByNode.type, exprString),
           isNullsFirst(orderByNode.null_order, exprString)});
    }
  }

  core::ExprPtr filter;
  if (functionExpr.filter) {
    filter = parseExpr(*functionExpr.filter, options);
  }

  auto* call = callExpr->as<core::CallExpr>();
  return std::make_shared<core::AggregateCallExpr>(
      call->name(),
      call->inputs(),
      functionExpr.distinct,
      std::move(filter),
      std::move(orderBy),
      callExpr->alias());
}

namespace {

using WindowType = core::WindowCallExpr::WindowType;
using BoundType = core::WindowCallExpr::BoundType;

WindowType parseWindowType(const WindowExpression& expr) {
  auto isRows = [](const WindowBoundary& boundary) {
    return boundary == WindowBoundary::CURRENT_ROW_ROWS ||
        boundary == WindowBoundary::EXPR_FOLLOWING_ROWS ||
        boundary == WindowBoundary::EXPR_PRECEDING_ROWS;
  };

  return (isRows(expr.start) || isRows(expr.end)) ? WindowType::kRows
                                                  : WindowType::kRange;
}

BoundType parseBoundType(WindowBoundary boundary) {
  switch (boundary) {
    case WindowBoundary::CURRENT_ROW_RANGE:
    case WindowBoundary::CURRENT_ROW_ROWS:
    case WindowBoundary::CURRENT_ROW_GROUPS:
      return BoundType::kCurrentRow;
    case WindowBoundary::EXPR_PRECEDING_ROWS:
    case WindowBoundary::EXPR_PRECEDING_RANGE:
    case WindowBoundary::EXPR_PRECEDING_GROUPS:
      return BoundType::kPreceding;
    case WindowBoundary::EXPR_FOLLOWING_ROWS:
    case WindowBoundary::EXPR_FOLLOWING_RANGE:
    case WindowBoundary::EXPR_FOLLOWING_GROUPS:
      return BoundType::kFollowing;
    case WindowBoundary::UNBOUNDED_FOLLOWING:
      return BoundType::kUnboundedFollowing;
    case WindowBoundary::UNBOUNDED_PRECEDING:
      return BoundType::kUnboundedPreceding;
    case WindowBoundary::INVALID:
      VELOX_UNREACHABLE();
  }
  VELOX_UNREACHABLE();
}

core::WindowCallExprPtr buildWindowCallExpr(
    ParsedExpression& parsedExpr,
    const std::string& windowString,
    const ParseOptions& options) {
  auto& windowExpr = dynamic_cast<WindowExpression&>(parsedExpr);

  std::vector<core::ExprPtr> partitionKeys;
  for (const auto& partition : windowExpr.partitions) {
    partitionKeys.push_back(parseExpr(*partition, options));
  }

  std::vector<core::SortKey> orderByKeys;
  for (const auto& orderByNode : windowExpr.orders) {
    orderByKeys.push_back(
        {parseExpr(*orderByNode.expression, options),
         isAscending(orderByNode.type, windowString),
         isNullsFirst(orderByNode.null_order, windowString)});
  }

  std::vector<core::ExprPtr> params;
  params.reserve(windowExpr.children.size());
  for (const auto& c : windowExpr.children) {
    params.emplace_back(parseExpr(*c, options));
  }

  // Lead and Lag functions have extra offset and default_value arguments.
  if (windowExpr.offset_expr) {
    params.emplace_back(parseExpr(*windowExpr.offset_expr, options));
  }
  if (windowExpr.default_expr) {
    params.emplace_back(parseExpr(*windowExpr.default_expr, options));
  }

  core::ExprPtr startValue;
  if (windowExpr.start_expr) {
    startValue = parseExpr(*windowExpr.start_expr, options);
  }
  core::ExprPtr endValue;
  if (windowExpr.end_expr) {
    endValue = parseExpr(*windowExpr.end_expr, options);
  }

  auto endType = parseBoundType(windowExpr.end);
  if (options.correctWindowFrameDefault && orderByKeys.empty() &&
      endType == core::WindowCallExpr::BoundType::kCurrentRow) {
    endType = core::WindowCallExpr::BoundType::kUnboundedFollowing;
  }

  return std::make_shared<core::WindowCallExpr>(
      normalizeFuncName(windowExpr.function_name),
      std::move(params),
      std::move(partitionKeys),
      std::move(orderByKeys),
      core::WindowCallExpr::Frame{
          parseWindowType(windowExpr),
          parseBoundType(windowExpr.start),
          std::move(startValue),
          endType,
          std::move(endValue)},
      windowExpr.ignore_nulls,
      getAlias(windowExpr));
}

} // namespace

core::WindowCallExprPtr parseWindowExpr(
    const std::string& windowString,
    const ParseOptions& options) {
  auto parsedExpr = parseSingleExpression(windowString);
  VELOX_CHECK(
      parsedExpr->IsWindow(),
      "Invalid window function expression: {}",
      windowString);

  return buildWindowCallExpr(*parsedExpr, windowString, options);
}

core::ExprPtr parseScalarOrWindowExpr(
    const std::string& exprString,
    const ParseOptions& options) {
  auto parsedExpr = parseSingleExpression(exprString);
  if (parsedExpr->IsWindow()) {
    return buildWindowCallExpr(*parsedExpr, exprString, options);
  }
  return parseExpr(*parsedExpr, options);
}

} // namespace facebook::velox::duckdb
