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

#include "velox/exec/fuzzer/ToSQLUtil.h"

namespace facebook::velox::exec::test {

std::string toCallSql(const core::CallTypedExprPtr& call);
std::string toCastSql(const core::CastTypedExprPtr& cast);
std::string toConcatSql(const core::ConcatTypedExprPtr& concat);

std::string escape(const std::string& input) {
  std::string result;
  result.reserve(input.size());
  for (auto i = 0; i < input.size(); ++i) {
    if (input[i] == '\'') {
      result.push_back('\'');
    }
    result.push_back(input[i]);
  }
  return result;
}

std::string toTypeSql(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ARRAY:
      return fmt::format("array({})", toTypeSql(type->childAt(0)));
    case TypeKind::MAP:
      return fmt::format(
          "map({}, {})",
          toTypeSql(type->childAt(0)),
          toTypeSql(type->childAt(1)));
    case TypeKind::ROW: {
      const auto& rowType = type->asRow();
      std::stringstream sql;
      sql << "row(";
      for (auto i = 0; i < type->size(); ++i) {
        appendComma(i, sql);
        sql << rowType.nameOf(i) << " ";
        sql << toTypeSql(type->childAt(i));
      }
      sql << ")";
      return sql.str();
    }
    default:
      if (type->isPrimitiveType()) {
        return type->toString();
      }
      VELOX_UNSUPPORTED("Type is not supported: {}", type->toString());
  }
}

std::string typedExprToSql(const core::TypedExprPtr& expr) {
  if (auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    return field->name();
  } else if (
      auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr)) {
    return toCallSql(call);
  } else if (
      auto cast = std::dynamic_pointer_cast<const core::CastTypedExpr>(expr)) {
    return toCastSql(cast);
  } else if (
      auto concat =
          std::dynamic_pointer_cast<const core::ConcatTypedExpr>(expr)) {
    return toConcatSql(concat);
  } else if (
      auto lambda =
          std::dynamic_pointer_cast<const core::LambdaTypedExpr>(expr)) {
    const auto& signature = lambda->signature();
    const auto& body =
        std::dynamic_pointer_cast<const core::CallTypedExpr>(lambda->body());
    VELOX_CHECK_NOT_NULL(body);

    std::stringstream sql;
    sql << "(";
    for (auto j = 0; j < signature->size(); ++j) {
      appendComma(j, sql);
      sql << signature->nameOf(j);
    }
    sql << ") -> " << toCallSql(body);
    return sql.str();
  } else if (
      auto constantArg =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(expr)) {
    std::stringstream sql;
    if (!constantArg->hasValueVector()) {
      sql << constantArg->toString();
    } else if (constantArg->type()->isVarchar()) {
      sql << "'" << escape(constantArg->valueVector()->toString(0)) << "'";
    } else {
      VELOX_NYI();
    }
    return sql.str();
  }
  VELOX_NYI();
}

void toCallInputsSql(
    const std::vector<core::TypedExprPtr>& inputs,
    std::stringstream& sql) {
  for (auto i = 0; i < inputs.size(); ++i) {
    appendComma(i, sql);

    const auto& input = inputs.at(i);
    /*if (auto field =
            std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                input)) {
      sql << field->name();
    } else if (
        auto call =
            std::dynamic_pointer_cast<const core::CallTypedExpr>(input)) {
      sql << toCallSql(call);
    } else if (
        auto lambda =
            std::dynamic_pointer_cast<const core::LambdaTypedExpr>(input)) {
      const auto& signature = lambda->signature();
      const auto& body =
          std::dynamic_pointer_cast<const core::CallTypedExpr>(lambda->body());
      VELOX_CHECK_NOT_NULL(body);

      sql << "(";
      for (auto j = 0; j < signature->size(); ++j) {
        appendComma(j, sql);
        sql << signature->nameOf(j);
      }

      sql << ") -> " << toCallSql(body);
    } else if (
        auto constantArg =
            std::dynamic_pointer_cast<const core::ConstantTypedExpr>(input)) {
      if (!constantArg->hasValueVector()) {
        sql << constantArg->toString();
      } else if (constantArg->type()->isVarchar()) {
        sql << "'" << escape(constantArg->valueVector()->toString(0)) << "'";
      } else {
        VELOX_NYI();
      }
    } else {
      VELOX_NYI();
    }*/
    sql << typedExprToSql(input);
  }
}

std::string toCallSql(const core::CallTypedExprPtr& call) {
  std::stringstream sql;
  sql << call->name() << "(";
  toCallInputsSql(call->inputs(), sql);
  sql << ")";
  return sql.str();
}

std::string toCastSql(const core::CastTypedExprPtr& cast) {
  std::stringstream sql;
  sql << "cast(";
  toCallInputsSql(cast->inputs(), sql);
  sql << " as " << toTypeSql(cast->type());
  sql << ")";
  return sql.str();
}

std::string toConcatSql(const core::ConcatTypedExprPtr& concat) {
  std::stringstream sql;
  sql << "concat(";
  toCallInputsSql(concat->inputs(), sql);
  sql << ")";
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

  if (!sortingKeys.empty()) {
    sql << " ORDER BY ";

    for (int i = 0; i < sortingKeys.size(); i++) {
      appendComma(i, sql);
      sql << sortingKeys[i]->name() << " " << sortingOrders[i].toString();
    }
  }

  sql << ")";
  return sql.str();
}

} // namespace facebook::velox::exec::test
