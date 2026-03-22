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

#include "velox/functions/prestosql/types/PrestoTypeSql.h"

#include <fmt/format.h>
#include <velox/common/base/Exceptions.h>
#include <iomanip>
#include <sstream>

namespace facebook::velox {

namespace {
bool isPlainArray(const Type& type) {
  return type.isArray() && typeid(type) == typeid(ArrayType);
}

bool isPlainMap(const Type& type) {
  return type.isMap() && typeid(type) == typeid(MapType);
}

bool isPlainRow(const Type& type) {
  return type.isRow() && typeid(type) == typeid(RowType);
}

// Returns true if a ROW field name needs quoting in Presto SQL.
// Field names that are not simple identifiers (letters, digits, underscores
// starting with a letter or underscore) must be quoted.
bool needsQuoting(const std::string& name) {
  if (name.empty()) {
    return false;
  }
  if (name[0] != '_' && !std::isalpha(name[0])) {
    return true;
  }
  for (auto c : name) {
    if (c != '_' && !std::isalnum(c)) {
      return true;
    }
  }
  return false;
}

} // namespace

std::string toPrestoTypeSql(const TypePtr& type) {
  if (isPlainArray(*type)) {
    return fmt::format("ARRAY({})", toPrestoTypeSql(type->childAt(0)));
  }

  if (isPlainMap(*type)) {
    return fmt::format(
        "MAP({}, {})",
        toPrestoTypeSql(type->childAt(0)),
        toPrestoTypeSql(type->childAt(1)));
  }

  if (isPlainRow(*type)) {
    const auto& rowType = type->asRow();
    std::stringstream sql;
    sql << "ROW(";
    for (auto i = 0; i < type->size(); ++i) {
      if (i > 0) {
        sql << ", ";
      }
      const auto& name = rowType.nameOf(i);
      if (!name.empty()) {
        if (needsQuoting(name)) {
          sql << std::quoted(name, '"', '"') << " ";
        } else {
          sql << name << " ";
        }
      }
      sql << toPrestoTypeSql(type->childAt(i));
    }
    sql << ")";
    return sql.str();
  }

  // Primitive types and custom types (e.g. IPPREFIX, BINGTILE).
  // For parameterized types (e.g. TDIGEST(DOUBLE), DECIMAL(10, 2)),
  // append parameters.
  const auto& params = type->parameters();
  if (params.empty()) {
    return type->name();
  }

  std::stringstream sql;
  sql << type->name() << "(";
  for (auto i = 0; i < params.size(); ++i) {
    if (i > 0) {
      sql << ", ";
    }
    switch (params[i].kind) {
      case TypeParameterKind::kType:
        sql << toPrestoTypeSql(params[i].type);
        break;
      case TypeParameterKind::kLongLiteral:
        sql << params[i].longLiteral.value();
        break;
      default:
        VELOX_UNSUPPORTED(
            "Unsupported type parameter kind: {}",
            TypeParameterKindName::toName(params[i].kind));
    }
  }
  sql << ")";
  return sql.str();
}

} // namespace facebook::velox
