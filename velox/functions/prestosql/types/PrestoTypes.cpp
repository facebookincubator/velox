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

#include "velox/functions/prestosql/types/PrestoTypes.h"

#include <fmt/format.h>
#include <velox/common/base/Exceptions.h>
#include <iomanip>
#include <sstream>
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

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

std::string PrestoTypes::toSql(const TypePtr& type) {
  if (isPlainArray(*type)) {
    return fmt::format("ARRAY({})", toSql(type->childAt(0)));
  }

  if (isPlainMap(*type)) {
    return fmt::format(
        "MAP({}, {})", toSql(type->childAt(0)), toSql(type->childAt(1)));
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
      sql << toSql(type->childAt(i));
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
        sql << toSql(params[i].type);
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

namespace {
template <TypeKind kind>
std::string scalarValueToString(
    const DecodedVector& decoded,
    vector_size_t index,
    const TypePtr& type) {
  using T = typename TypeTraits<kind>::NativeType;
  return PrestoTypes::valueToString(decoded.valueAt<T>(index), type);
}
} // namespace

std::string PrestoTypes::valueToString(
    const DecodedVector& decoded,
    vector_size_t index,
    const TypePtr& type) {
  if (decoded.isNullAt(index)) {
    return "null";
  }

  if (isIPPrefixType(type)) {
    auto* rowVector = decoded.base()->as<RowVector>();
    const auto decodedIndex = decoded.index(index);
    const auto ip =
        rowVector->childAt(0)->as<SimpleVector<int128_t>>()->valueAt(
            decodedIndex);
    const auto prefixLength =
        rowVector->childAt(1)->as<SimpleVector<int8_t>>()->valueAt(
            decodedIndex);
    char buffer[IPPrefixType::kMaxStringSize];
    return std::string(IPPREFIX()->valueToString(ip, prefixLength, buffer));
  }

  if (type->isPrimitiveType()) {
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        scalarValueToString, type->kind(), decoded, index, type);
  }

  const auto decodedIndex = decoded.index(index);

  if (type->isArray()) {
    auto* arrayVector = decoded.base()->as<ArrayVector>();
    const auto offset = arrayVector->offsetAt(decodedIndex);
    const auto size = arrayVector->sizeAt(decodedIndex);
    const auto& elements = arrayVector->elements();
    DecodedVector decodedElements(*elements);
    std::stringstream result;
    result << "[";
    for (auto i = 0; i < size; ++i) {
      if (i > 0) {
        result << ", ";
      }
      result << valueToString(decodedElements, offset + i, type->childAt(0));
    }
    result << "]";
    return result.str();
  }

  if (type->isMap()) {
    auto* mapVector = decoded.base()->as<MapVector>();
    const auto offset = mapVector->offsetAt(decodedIndex);
    const auto size = mapVector->sizeAt(decodedIndex);
    const auto& keys = mapVector->mapKeys();
    const auto& values = mapVector->mapValues();
    DecodedVector decodedKeys(*keys);
    DecodedVector decodedValues(*values);
    std::stringstream result;
    result << "{";
    for (auto i = 0; i < size; ++i) {
      if (i > 0) {
        result << ", ";
      }
      result << valueToString(decodedKeys, offset + i, type->childAt(0)) << "="
             << valueToString(decodedValues, offset + i, type->childAt(1));
    }
    result << "}";
    return result.str();
  }

  if (type->isRow()) {
    auto* rowVector = decoded.base()->as<RowVector>();
    const auto& rowType = type->asRow();
    std::stringstream result;
    result << "{";
    for (auto i = 0; i < rowVector->childrenSize(); ++i) {
      if (i > 0) {
        result << ", ";
      }
      const auto& child = rowVector->childAt(i);
      DecodedVector decodedChild(*child);
      const auto& fieldName = rowType.nameOf(i);
      if (fieldName.empty()) {
        result << "field" << i;
      } else {
        result << fieldName;
      }
      result << "="
             << valueToString(decodedChild, decodedIndex, type->childAt(i));
    }
    result << "}";
    return result.str();
  }

  return decoded.toString(index);
}

std::string PrestoTypes::timestampToPrestoString(Timestamp value) {
  // Uses millisecond precision to match Presto Java's SqlTimestamp.JSON_FORMAT
  // ("uuuu-MM-dd HH:mm:ss.SSS"). Sub-millisecond nanos are truncated.
  // This is consistent with PrestoCastHooks (non-legacy mode), which sets the
  // same options for CAST(timestamp AS varchar).
  // TODO: Support microsecond precision for TIMESTAMP(6) if needed.
  static const TimestampToStringOptions kPrestoOptions{
      .precision = TimestampToStringOptions::Precision::kMilliseconds,
      .zeroPaddingYear = true,
      .dateTimeSeparator = ' ',
  };
  return value.toString(kPrestoOptions);
}

std::string PrestoTypes::toHex(StringView value) {
  std::string result;
  result.reserve(value.size() * 3);
  for (size_t i = 0; i < value.size(); ++i) {
    if (i > 0) {
      result += ' ';
    }
    fmt::format_to(
        std::back_inserter(result),
        "{:02x}",
        static_cast<uint8_t>(value.data()[i]));
  }
  return result;
}

} // namespace facebook::velox
