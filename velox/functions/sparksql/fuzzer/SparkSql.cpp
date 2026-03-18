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

#include "velox/functions/sparksql/fuzzer/SparkSql.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::functions::sparksql::fuzzer {
namespace {

std::string toDereferenceSql(const core::DereferenceTypedExpr& dereference) {
  std::stringstream sql;
  toCallInputsSql(dereference.inputs(), sql);
  sql << "." << dereference.name();
  return sql.str();
}

template <typename T>
T getConstantValue(const core::ConstantTypedExpr& expr) {
  if (expr.hasValueVector()) {
    return expr.valueVector()->as<SimpleVector<T>>()->valueAt(0);
  } else {
    return expr.value().value<T>();
  }
}

template <>
std::string getConstantValue(const core::ConstantTypedExpr& expr) {
  if (expr.hasValueVector()) {
    return expr.valueVector()
        ->as<SimpleVector<StringView>>()
        ->valueAt(0)
        .getString();
  } else {
    return expr.value().value<std::string>();
  }
}

// Returns a mapping from Velox function names to the corresponding unary
// operators supported in Spark SQL.
const std::unordered_map<std::string, std::string>& unaryOperatorMap() {
  static std::unordered_map<std::string, std::string> unaryOperatorMap{
      {"unaryminus", "-"},
      {"not", "not"},
      {"bitwise_not", "~"},
  };
  return unaryOperatorMap;
}

// Returns a mapping from Velox function names to the corresponding binary
// operators supported in Spark SQL.
const std::unordered_map<std::string, std::string>& binaryOperatorMap() {
  static std::unordered_map<std::string, std::string> binaryOperatorMap{
      {"add", "+"},
      {"subtract", "-"},
      {"multiply", "*"},
      {"divide", "/"},
      {"remainder", "%"},
      {"equalto", "="},
      {"lessthan", "<"},
      {"greaterthan", ">"},
      {"lessthanorequal", "<="},
      {"greaterthanorequal", ">="},
      {"bitwise_and", "&"},
      {"bitwise_or", "|"},
      {"bitwise_xor", "^"},
      {"equalnullsafe", "<=>"},
  };
  return binaryOperatorMap;
}

// Returns a mapping from Velox function names to the corresponding function
// names in Spark SQL.
const std::unordered_map<std::string, std::string>& sparkFunctionMap() {
  static std::unordered_map<std::string, std::string> sparkFunctionMap{
      {"checked_add", "try_add"},
      {"checked_subtract", "try_subtract"},
      {"checked_multiply", "try_multiply"},
      {"checked_divide", "try_divide"},
      {"doy", "dayofyear"},
      {"to_unix_timestamp", "unix_timestamp"},
      {"week_of_year", "week"},
      {"year_of_week", "yearofweek"},
  };
  return sparkFunctionMap;
}

} // namespace

void appendComma(int32_t i, std::stringstream& sql) {
  if (i > 0) {
    sql << ", ";
  }
}

// Returns the SQL string of the given type.
std::string toTypeSql(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ARRAY:
      return fmt::format("ARRAY<{}>", toTypeSql(type->childAt(0)));
    case TypeKind::MAP:
      return fmt::format(
          "MAP<{}, {}>",
          toTypeSql(type->childAt(0)),
          toTypeSql(type->childAt(1)));
    case TypeKind::ROW: {
      const auto& rowType = type->asRow();
      std::stringstream sql;
      sql << "STRUCT<";
      for (auto i = 0; i < type->size(); ++i) {
        appendComma(i, sql);
        sql << rowType.nameOf(i) << ":" << toTypeSql(type->childAt(i));
      }
      sql << ">";
      return sql.str();
    }
    case TypeKind::INTEGER: {
      if (type->isDate()) {
        return "DATE";
      }
      return "INT";
    }
    case TypeKind::BIGINT: {
      if (type->isDecimal()) {
        return type->toString();
      }
      return "BIGINT";
    }
    case TypeKind::HUGEINT:
      VELOX_UNSUPPORTED("Type is not supported: {}", type->toString());
    case TypeKind::REAL:
      return "FLOAT";
    case TypeKind::VARCHAR:
      return "STRING";
    case TypeKind::VARBINARY:
      return "BINARY";
    case TypeKind::UNKNOWN:
      VELOX_UNSUPPORTED("Type is not supported: {}", type->toString());
    default: {
      if (type->isPrimitiveType()) {
        return type->name();
      }
      VELOX_UNSUPPORTED("Type is not supported: {}", type->toString());
    }
  }
}

std::string toLambdaSql(const core::LambdaTypedExprPtr& lambda) {
  std::stringstream sql;
  const auto& signature = lambda->signature();

  sql << "(";
  for (auto j = 0; j < signature->size(); ++j) {
    appendComma(j, sql);
    sql << signature->nameOf(j);
  }

  sql << ") -> ";
  toCallInputsSql({lambda->body()}, sql);
  return sql.str();
}

void toCallInputsSql(
    const std::vector<core::TypedExprPtr>& inputs,
    std::stringstream& sql) {
  for (auto i = 0; i < inputs.size(); ++i) {
    appendComma(i, sql);

    const auto& input = inputs.at(i);
    if (auto field =
            std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                input)) {
      if (field->isInputColumn()) {
        sql << field->name();
      } else {
        toCallInputsSql(field->inputs(), sql);
        sql << fmt::format(".{}", field->name());
      }
    } else if (
        auto call =
            std::dynamic_pointer_cast<const core::CallTypedExpr>(input)) {
      sql << toCallSql(call);
    } else if (
        auto lambda =
            std::dynamic_pointer_cast<const core::LambdaTypedExpr>(input)) {
      sql << toLambdaSql(lambda);
    } else if (
        auto constantArg =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(input)) {
      sql << toConstantSql(*constantArg);
    } else if (
        auto castArg =
            std::dynamic_pointer_cast<const core::CastTypedExpr>(input)) {
      sql << toCastSql(*castArg);
    } else if (
        auto concatArg =
            std::dynamic_pointer_cast<const core::ConcatTypedExpr>(input)) {
      sql << toConcatSql(*concatArg);
    } else if (
        auto dereferenceArg =
            std::dynamic_pointer_cast<const core::DereferenceTypedExpr>(
                input)) {
      sql << toDereferenceSql(*dereferenceArg);
    } else {
      VELOX_NYI("Unsupported input expression: {}.", input->toString());
    }
  }
}

std::string toCallSql(const core::CallTypedExprPtr& call) {
  std::stringstream sql;
  // Some functions require special SQL syntax, so handle them first.
  const auto& unaryOperators = unaryOperatorMap();
  const auto& binaryOperators = binaryOperatorMap();
  const auto& functionMap = sparkFunctionMap();
  if (unaryOperators.count(call->name()) > 0) {
    VELOX_CHECK_EQ(
        call->inputs().size(), 1, "Expected one argument to a unary operator");
    sql << "(";
    sql << fmt::format("{} ", unaryOperators.at(call->name()));
    toCallInputsSql({call->inputs()[0]}, sql);
    sql << ")";
  } else if (binaryOperators.count(call->name()) > 0) {
    VELOX_CHECK_EQ(
        call->inputs().size(),
        2,
        "Expected two arguments to a binary operator");
    sql << "(";
    toCallInputsSql({call->inputs()[0]}, sql);
    sql << fmt::format(" {} ", binaryOperators.at(call->name()));
    toCallInputsSql({call->inputs()[1]}, sql);
    sql << ")";
  } else if (call->name() == "isnull" || call->name() == "isnotnull") {
    sql << "(";
    VELOX_CHECK_EQ(
        call->inputs().size(),
        1,
        "Expected one argument to function 'isnull' or 'isnotnull'");
    toCallInputsSql({call->inputs()[0]}, sql);
    sql << fmt::format(" is{} null", call->name() == "isnotnull" ? " not" : "");
    sql << ")";
  } else if (call->name() == "in") {
    VELOX_CHECK_GE(
        call->inputs().size(),
        2,
        "Expected at least two arguments to function 'in'");
    toCallInputsSql({call->inputs()[0]}, sql);
    sql << " in (";
    for (auto i = 1; i < call->inputs().size(); ++i) {
      appendComma(i - 1, sql);
      toCallInputsSql({call->inputs()[i]}, sql);
    }
    sql << ")";
  } else if (call->name() == "like") {
    VELOX_CHECK_GE(
        call->inputs().size(),
        2,
        "Expected at least two arguments to function 'like'");
    VELOX_CHECK_LE(
        call->inputs().size(),
        3,
        "Expected at most three arguments to function 'like'");
    sql << "(";
    toCallInputsSql({call->inputs()[0]}, sql);
    sql << " like ";
    toCallInputsSql({call->inputs()[1]}, sql);
    if (call->inputs().size() == 3) {
      sql << " escape ";
      toCallInputsSql({call->inputs()[2]}, sql);
    }
    sql << ")";
  } else if (call->name() == "or" || call->name() == "and") {
    VELOX_CHECK_GE(
        call->inputs().size(),
        2,
        "Expected at least two arguments to function 'or' or 'and'");
    sql << "(";
    const auto& inputs = call->inputs();
    for (auto i = 0; i < inputs.size(); ++i) {
      if (i > 0) {
        sql << fmt::format(" {} ", call->name());
      }
      toCallInputsSql({inputs[i]}, sql);
    }
    sql << ")";
  } else if (call->name() == "array") {
    sql << "ARRAY(";
    toCallInputsSql(call->inputs(), sql);
    sql << ")";
  } else if (call->name() == "row_constructor") {
    VELOX_CHECK_GE(
        call->inputs().size(),
        1,
        "Expected at least one argument to function 'row_constructor'");
    sql << "STRUCT(";
    toCallInputsSql(call->inputs(), sql);
    sql << ")";
  } else if (call->name() == "between") {
    VELOX_CHECK_EQ(
        call->inputs().size(),
        3,
        "Expected three arguments to function 'between'");
    sql << "(";
    const auto& inputs = call->inputs();
    toCallInputsSql({inputs[0]}, sql);
    sql << " between ";
    toCallInputsSql({inputs[1]}, sql);
    sql << " and ";
    toCallInputsSql({inputs[2]}, sql);
    sql << ")";
  } else if (call->name() == "week_of_year" || call->name() == "year_of_week") {
    // Special handling for extract functions.
    VELOX_CHECK_EQ(
        call->inputs().size(), 1, "Expected one argument to extract function");
    sql << "extract('";
    if (functionMap.count(call->name()) > 0) {
      // If the Velox function name is different from the Spark SQL
      // function name, use the Spark SQL function name.
      sql << functionMap.at(call->name());
    } else {
      sql << call->name();
    }
    sql << "', ";
    toCallInputsSql({call->inputs()[0]}, sql);
    sql << ")";
  } else if (
      call->name() == "unix_seconds" || call->name() == "unix_millis" ||
      call->name() == "unix_micros") {
    VELOX_CHECK_EQ(
        call->inputs().size(), 1, "Expected one argument to unix functions");
    // Cast the input as timestamp to avoid timestamp_ntz being used in Spark.
    sql << call->name() << "(cast(";
    toCallInputsSql({call->inputs()[0]}, sql);
    sql << " as timestamp))";
  } else if (call->name() == "switch") {
    VELOX_CHECK_GE(
        call->inputs().size(),
        2,
        "Expected at least two arguments to function 'switch'");
    sql << "case";
    int i = 0;
    for (; i < call->inputs().size() - 1; i += 2) {
      sql << " when ";
      toCallInputsSql({call->inputs()[i]}, sql);
      sql << " then ";
      toCallInputsSql({call->inputs()[i + 1]}, sql);
    }
    if (i < call->inputs().size()) {
      sql << " else ";
      toCallInputsSql({call->inputs()[i]}, sql);
    }
    sql << " end";
  } else {
    // Regular function call syntax.
    if (functionMap.count(call->name()) > 0) {
      // If the Velox function name is different from the Spark SQL
      // function name, use the Spark SQL function name.
      sql << functionMap.at(call->name());
    } else {
      sql << call->name();
    }
    sql << "(";
    toCallInputsSql(call->inputs(), sql);
    sql << ")";
  }
  return sql.str();
}

std::string toCastSql(const core::CastTypedExpr& cast) {
  std::stringstream sql;
  if (cast.isTryCast()) {
    sql << "try_cast(";
  } else {
    sql << "cast(";
  }
  toCallInputsSql(cast.inputs(), sql);
  sql << " as " << toTypeSql(cast.type());
  sql << ")";
  return sql.str();
}

std::string toConcatSql(const core::ConcatTypedExpr& concat) {
  std::stringstream input;
  toCallInputsSql(concat.inputs(), input);
  return fmt::format(
      "cast(struct({}) as {})", input.str(), toTypeSql(concat.type()));
}

std::string toConstantSql(const core::ConstantTypedExpr& constant) {
  const auto& type = constant.type();
  const auto typeSql = toTypeSql(type);

  std::stringstream sql;
  if (constant.isNull()) {
    // Syntax like BIGINT('null') for typed null is not supported, so use cast
    // instead.
    sql << fmt::format("cast(null as {})", typeSql);
  } else if (type->isVarchar() || type->isVarbinary()) {
    // Escape single quote in string literals used in SQL texts.
    auto constantValue = getConstantValue<std::string>(constant);
    auto quoted = std::quoted(constantValue, '\'', '\'');
    if (type->isVarbinary()) {
      sql << typeSql << "(" << quoted << ")";
    } else {
      sql << quoted;
    }
  } else if (type->isDecimal()) {
    sql << fmt::format("cast('{}' as {})", constant.toString(), typeSql);
  } else if (type->isPrimitiveType()) {
    sql << fmt::format("{}('{}')", typeSql, constant.toString());
  } else {
    VELOX_NYI(
        "Constant expressions of {} are not supported yet.", type->toString());
  }
  return sql.str();
}

std::string toAggregateCallSql(
    const core::CallTypedExprPtr& call,
    const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders,
    bool distinct) {
  VELOX_CHECK_EQ(sortingKeys.size(), sortingOrders.size());
  std::stringstream sql;
  sql << call->name() << "(";

  if (distinct) {
    sql << "distinct ";
  }

  toCallInputsSql(call->inputs(), sql);

  sql << ")";
  return sql.str();
}

} // namespace facebook::velox::functions::sparksql::fuzzer
