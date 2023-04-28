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
#include "velox/duckdb/conversion/DuckConversion.h"
#include "velox/type/Variant.h"

namespace facebook::velox::duckdb {
using ::duckdb::DataChunk;
using ::duckdb::Date;
using ::duckdb::LogicalType;
using ::duckdb::LogicalTypeId;
using ::duckdb::Value;
using ::duckdb::Vector;
using ::duckdb::VectorOperations;
using ::duckdb::VectorType;

using ::duckdb::data_ptr_t;
using ::duckdb::date_t;
using ::duckdb::dtime_t;
using ::duckdb::string_t;
using ::duckdb::timestamp_t;

variant decimalVariant(const Value& val) {
  VELOX_DCHECK(val.type().id() == LogicalTypeId::DECIMAL)
  uint8_t precision;
  uint8_t scale;
  val.type().GetDecimalProperties(precision, scale);
  auto decimalType = DECIMAL(precision, scale);
  switch (val.type().InternalType()) {
    case ::duckdb::PhysicalType::INT128: {
      auto unscaledValue = val.GetValueUnsafe<::duckdb::hugeint_t>();
      return variant::longDecimal(
          buildInt128(unscaledValue.upper, unscaledValue.lower), decimalType);
    }
    case ::duckdb::PhysicalType::INT16: {
      return variant::shortDecimal(val.GetValueUnsafe<int16_t>(), decimalType);
    }
    case ::duckdb::PhysicalType::INT32: {
      return variant::shortDecimal(val.GetValueUnsafe<int32_t>(), decimalType);
    }
    case ::duckdb::PhysicalType::INT64: {
      return variant::shortDecimal(val.GetValueUnsafe<int64_t>(), decimalType);
    }
    default:
      VELOX_UNSUPPORTED();
  }
}

//! Type mapping for velox -> DuckDB conversions
LogicalType fromVeloxType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return LogicalType::BOOLEAN;
    case TypeKind::TINYINT:
      return LogicalType::TINYINT;
    case TypeKind::SMALLINT:
      return LogicalType::SMALLINT;
    case TypeKind::INTEGER:
      return LogicalType::INTEGER;
    case TypeKind::BIGINT:
      if (isIntervalDayTimeType(type)) {
        return LogicalType::INTERVAL;
      }
      return LogicalType::BIGINT;
    case TypeKind::REAL:
      return LogicalType::FLOAT;
    case TypeKind::DOUBLE:
      return LogicalType::DOUBLE;
    case TypeKind::VARCHAR:
      return LogicalType::VARCHAR;
    case TypeKind::TIMESTAMP:
      return LogicalType::TIMESTAMP;
    case TypeKind::DATE:
      return LogicalType::DATE;
    case TypeKind::LONG_DECIMAL:
      return LogicalType::DECIMAL(
          type->asLongDecimal().precision(), type->asLongDecimal().scale());
    case TypeKind::SHORT_DECIMAL:
      return LogicalType::DECIMAL(
          type->asShortDecimal().precision(), type->asShortDecimal().scale());
    case TypeKind::ARRAY:
      return LogicalType::LIST(fromVeloxType(type->childAt(0)));
    case TypeKind::MAP:
      return LogicalType::MAP(
          fromVeloxType(type->childAt(0)), fromVeloxType(type->childAt(1)));
    case TypeKind::ROW: {
      const auto& rowType = type->asRow();
      std::vector<std::pair<std::string, LogicalType>> children;
      for (auto i = 0; i < rowType.size(); ++i) {
        children.push_back(
            {rowType.nameOf(i), fromVeloxType(rowType.childAt(i))});
      }
      return LogicalType::STRUCT(std::move(children));
    }
    default:
      throw std::runtime_error(
          "Unsupported type for velox -> DuckDB conversion: " +
          type->toString());
  }
}

//! Type mapping for DuckDB -> velox conversions, we support more types here
TypePtr toVeloxType(LogicalType type) {
  switch (type.id()) {
    case LogicalTypeId::SQLNULL:
      return UNKNOWN();
    case LogicalTypeId::BOOLEAN:
      return BOOLEAN();
    case LogicalTypeId::TINYINT:
      return TINYINT();
    case LogicalTypeId::SMALLINT:
      return SMALLINT();
    case LogicalTypeId::INTEGER:
      return INTEGER();
    case LogicalTypeId::BIGINT:
      return BIGINT();
    case LogicalTypeId::FLOAT:
      return REAL();
    case LogicalTypeId::DECIMAL:
      uint8_t width;
      uint8_t scale;
      type.GetDecimalProperties(width, scale);
      return DECIMAL(width, scale);
    case LogicalTypeId::HUGEINT:
    case LogicalTypeId::DOUBLE:
      return DOUBLE();
    case LogicalTypeId::VARCHAR:
      return VARCHAR();
    case LogicalTypeId::DATE:
      return DATE();
    case LogicalTypeId::TIMESTAMP:
      return TIMESTAMP();
    case LogicalTypeId::INTERVAL:
      return INTERVAL_DAY_TIME();
    case LogicalTypeId::BLOB:
      return VARBINARY();
    case LogicalTypeId::LIST: {
      auto childType = ::duckdb::ListType::GetChildType(type);
      return ARRAY(toVeloxType(childType));
    }
    case LogicalTypeId::MAP: {
      auto keyType = ::duckdb::MapType::KeyType(type);
      auto valueType = ::duckdb::MapType::ValueType(type);
      return MAP(toVeloxType(keyType), toVeloxType(valueType));
    }
    case LogicalTypeId::STRUCT: {
      std::vector<std::string> names;
      std::vector<TypePtr> types;

      auto numChildren = ::duckdb::StructType::GetChildCount(type);
      names.reserve(numChildren);
      types.reserve(numChildren);

      for (auto i = 0; i < numChildren; ++i) {
        names.push_back(::duckdb::StructType::GetChildName(type, i));
        types.push_back(
            toVeloxType(::duckdb::StructType::GetChildType(type, i)));
      }
      return ROW(std::move(names), std::move(types));
    }
    default:
      throw std::runtime_error(
          "unsupported type for duckdb -> velox conversion: " +
          type.ToString());
  }
}

variant duckValueToVariant(const Value& val) {
  switch (val.type().id()) {
    case LogicalTypeId::SQLNULL:
      return variant(TypeKind::UNKNOWN);
    case LogicalTypeId::BOOLEAN:
      return variant(val.GetValue<bool>());
    case LogicalTypeId::TINYINT:
      return variant(val.GetValue<int8_t>());
    case LogicalTypeId::SMALLINT:
      return variant(val.GetValue<int16_t>());
    case LogicalTypeId::INTEGER:
      return variant(val.GetValue<int32_t>());
    case LogicalTypeId::BIGINT:
      return variant(val.GetValue<int64_t>());
    case LogicalTypeId::FLOAT:
      return variant(val.GetValue<float>());
    case LogicalTypeId::DOUBLE:
      return variant(val.GetValue<double>());
    case LogicalTypeId::DECIMAL:
      return decimalVariant(val);
    case LogicalTypeId::VARCHAR:
      return variant(val.GetValue<std::string>());
    case LogicalTypeId::BLOB:
      return variant::binary(val.GetValue<std::string>());
    default:
      throw std::runtime_error(
          "unsupported type for duckdb value -> velox  variant conversion: " +
          val.type().ToString());
  }
}

std::string makeCreateTableSql(
    const std::string& tableName,
    const RowType& rowType) {
  std::ostringstream sql;
  sql << "CREATE TABLE " << tableName << "(";
  for (int32_t i = 0; i < rowType.size(); i++) {
    if (i > 0) {
      sql << ", ";
    }
    sql << rowType.nameOf(i) << " ";
    toTypeSql(rowType.childAt(i), sql);
  }
  sql << ")";
  return sql.str();
}

} // namespace facebook::velox::duckdb
