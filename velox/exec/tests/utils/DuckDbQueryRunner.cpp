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
#include "velox/exec/tests/utils/DuckDbQueryRunner.h"

#include "duckdb/common/types.hpp" // @manual
#include "velox/duckdb/conversion/DuckConversion.h"
#include "velox/vector/SimpleVector.h"
#include "velox/vector/VectorTypeUtils.h"

using facebook::velox::duckdb::duckdbTimestampToVelox;
using facebook::velox::duckdb::veloxTimestampToDuckDB;

namespace facebook::velox::exec::test {
namespace {

template <TypeKind kind>
::duckdb::Value duckValueAt(const VectorPtr& vector, vector_size_t index) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  return ::duckdb::Value(vector->as<SimpleVector<T>>()->valueAt(index));
}

// Forward declarations of the complex-type specializations. Nested complex
// types dispatch through these before their definitions appear below, so the
// declarations must precede any such instantiation.
template <>
::duckdb::Value duckValueAt<TypeKind::ARRAY>(
    const VectorPtr& vector,
    int32_t row);
template <>
::duckdb::Value duckValueAt<TypeKind::ROW>(
    const VectorPtr& vector,
    int32_t row);
template <>
::duckdb::Value duckValueAt<TypeKind::MAP>(
    const VectorPtr& vector,
    int32_t row);

template <>
::duckdb::Value duckValueAt<TypeKind::TINYINT>(
    const VectorPtr& vector,
    vector_size_t index) {
  return ::duckdb::Value::TINYINT(
      vector->as<SimpleVector<int8_t>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::SMALLINT>(
    const VectorPtr& vector,
    vector_size_t index) {
  return ::duckdb::Value::SMALLINT(
      vector->as<SimpleVector<int16_t>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::BOOLEAN>(
    const VectorPtr& vector,
    vector_size_t index) {
  return ::duckdb::Value::BOOLEAN(
      vector->as<SimpleVector<bool>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::VARCHAR>(
    const VectorPtr& vector,
    vector_size_t index) {
  // DuckDB requires zero-ending string
  auto copy = vector->as<SimpleVector<StringView>>()->valueAt(index).str();
  return ::duckdb::Value(copy);
}

template <>
::duckdb::Value duckValueAt<TypeKind::VARBINARY>(
    const VectorPtr& vector,
    vector_size_t index) {
  // DuckDB requires zero-ending string
  auto stringView = vector->as<SimpleVector<StringView>>()->valueAt(index);
  return ::duckdb::Value::BLOB(
      reinterpret_cast<const uint8_t*>(stringView.begin()), stringView.size());
}

template <>
::duckdb::Value duckValueAt<TypeKind::TIMESTAMP>(
    const VectorPtr& vector,
    vector_size_t index) {
  using T = typename KindToFlatVector<TypeKind::TIMESTAMP>::WrapperType;
  return ::duckdb::Value::TIMESTAMP(
      veloxTimestampToDuckDB(vector->as<SimpleVector<T>>()->valueAt(index)));
}

template <>
::duckdb::Value duckValueAt<TypeKind::INTEGER>(
    const VectorPtr& vector,
    vector_size_t index) {
  auto type = vector->type();
  if (type->isDate()) {
    return ::duckdb::Value::DATE(
        ::duckdb::Date::EpochDaysToDate(
            vector->as<SimpleVector<int32_t>>()->valueAt(index)));
  }
  return ::duckdb::Value(vector->as<SimpleVector<int32_t>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::BIGINT>(
    const VectorPtr& vector,
    vector_size_t index) {
  using T = typename KindToFlatVector<TypeKind::BIGINT>::WrapperType;
  auto type = vector->type();
  if (type->isShortDecimal()) {
    const auto& decimalType = type->asShortDecimal();
    return ::duckdb::Value::DECIMAL(
        vector->as<SimpleVector<T>>()->valueAt(index),
        decimalType.precision(),
        decimalType.scale());
  }

  if (type->isIntervalDayTime()) {
    static constexpr int64_t kMicrosecondsInDay =
        1000L * 1000L * 60L * 60L * 24L;
    const auto interval = vector->as<SimpleVector<int64_t>>()->valueAt(index);
    const int64_t microseconds = interval % kMicrosecondsInDay;
    const int64_t days = interval / kMicrosecondsInDay;
    return ::duckdb::Value::INTERVAL(0, days, microseconds);
  }

  if (type->isTime()) {
    VELOX_DCHECK(type->equivalent(*TIME()));
    // TIME is stored as milliseconds since midnight in Velox.
    // DuckDB TIME is stored as microseconds since midnight.
    const auto timeMillis = vector->as<SimpleVector<int64_t>>()->valueAt(index);
    return ::duckdb::Value::TIME(::duckdb::dtime_t(timeMillis * 1000L));
  }

  return ::duckdb::Value(vector->as<SimpleVector<T>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::HUGEINT>(
    const VectorPtr& vector,
    vector_size_t index) {
  using T = typename KindToFlatVector<TypeKind::HUGEINT>::WrapperType;
  auto val = vector->as<SimpleVector<T>>()->valueAt(index);
  auto duckVal = ::duckdb::hugeint_t();
  duckVal.lower = (val << 64) >> 64;
  duckVal.upper = (val >> 64);
  if (vector->type()->isLongDecimal()) {
    auto type = vector->type()->asLongDecimal();
    return ::duckdb::Value::DECIMAL(
        std::move(duckVal), type.precision(), type.scale());
  }
  // Flat vector is HUGEINT type and not the logical decimal type.
  return ::duckdb::Value::HUGEINT(std::move(duckVal));
}

template <>
::duckdb::Value duckValueAt<TypeKind::ARRAY>(
    const VectorPtr& vector,
    int32_t row) {
  auto arrayVector = vector->wrappedVector()->as<ArrayVector>();
  auto arrayRow = vector->wrappedIndex(row);
  auto& elements = arrayVector->elements();
  auto offset = arrayVector->offsetAt(arrayRow);
  auto size = arrayVector->sizeAt(arrayRow);

  if (size == 0) {
    return ::duckdb::Value::EMPTYLIST(duckdb::fromVeloxType(elements->type()));
  }

  ::duckdb::vector<::duckdb::Value> array;
  array.reserve(size);
  for (auto i = 0; i < size; i++) {
    auto innerRow = offset + i;
    if (elements->isNullAt(innerRow)) {
      array.emplace_back(
          ::duckdb::Value(duckdb::fromVeloxType(elements->type())));
    } else {
      array.emplace_back(VELOX_DYNAMIC_TYPE_DISPATCH(
          duckValueAt, elements->typeKind(), elements, innerRow));
    }
  }

  return ::duckdb::Value::LIST(array);
}

template <>
::duckdb::Value duckValueAt<TypeKind::ROW>(
    const VectorPtr& vector,
    int32_t row) {
  auto rowVector = vector->wrappedVector()->as<RowVector>();
  auto rowRow = vector->wrappedIndex(row);
  auto rowType = asRowType(rowVector->type());

  ::duckdb::vector<std::pair<std::string, ::duckdb::Value>> fields;
  for (auto i = 0; i < rowType->size(); ++i) {
    if (rowVector->childAt(i)->isNullAt(rowRow)) {
      fields.push_back({rowType->nameOf(i), ::duckdb::Value(nullptr)});
    } else {
      fields.push_back(
          {rowType->nameOf(i),
           VELOX_DYNAMIC_TYPE_DISPATCH(
               duckValueAt,
               rowType->childAt(i)->kind(),
               rowVector->childAt(i),
               rowRow)});
    }
  }

  return ::duckdb::Value::STRUCT(fields);
}

template <>
::duckdb::Value duckValueAt<TypeKind::MAP>(
    const VectorPtr& vector,
    int32_t row) {
  auto mapVector = vector->wrappedVector()->as<MapVector>();
  auto mapRow = vector->wrappedIndex(row);
  const auto& mapKeys = mapVector->mapKeys();
  const auto& mapValues = mapVector->mapValues();
  auto offset = mapVector->offsetAt(mapRow);
  auto size = mapVector->sizeAt(mapRow);
  auto mapType = ::duckdb::ListType::GetChildType(
      ::duckdb::LogicalType::MAP(
          duckdb::fromVeloxType(mapKeys->type()),
          duckdb::fromVeloxType(mapValues->type())));
  if (size == 0) {
    return ::duckdb::Value::MAP(mapType, ::duckdb::vector<::duckdb::Value>());
  }

  ::duckdb::vector<::duckdb::Value> duckMap;
  for (auto i = 0; i < size; i++) {
    auto innerRow = offset + i;
    auto key = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        duckValueAt, mapKeys->typeKind(), mapKeys, innerRow);
    ::duckdb::Value value;
    if (mapValues->isNullAt(innerRow)) {
      value = ::duckdb::Value(duckdb::fromVeloxType(mapValues->type()));
    } else {
      value = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          duckValueAt, mapValues->typeKind(), mapValues, innerRow);
    }
    ::duckdb::child_list_t<::duckdb::Value> mapStruct;
    mapStruct.push_back(make_pair("key", key));
    mapStruct.push_back(make_pair("value", value));
    duckMap.push_back(::duckdb::Value::STRUCT(mapStruct));
  }
  return ::duckdb::Value::MAP(mapType, duckMap);
}

template <TypeKind kind>
Variant variantAt(::duckdb::DataChunk* dataChunk, int32_t row, int32_t column) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  return Variant(dataChunk->GetValue(column, row).GetValue<T>());
}

template <>
Variant variantAt<TypeKind::VARCHAR>(
    ::duckdb::DataChunk* dataChunk,
    int32_t row,
    int32_t column) {
  return Variant(
      StringView(::duckdb::StringValue::Get(dataChunk->GetValue(column, row))));
}

template <>
Variant variantAt<TypeKind::HUGEINT>(
    ::duckdb::DataChunk* dataChunk,
    int32_t row,
    int32_t column) {
  auto unscaledValue =
      dataChunk->GetValue(column, row).GetValue<::duckdb::hugeint_t>();
  return Variant(HugeInt::build(unscaledValue.upper, unscaledValue.lower));
}

template <>
Variant variantAt<TypeKind::VARBINARY>(
    ::duckdb::DataChunk* dataChunk,
    int32_t row,
    int32_t column) {
  return Variant::binary(
      std::string(
          ::duckdb::StringValue::Get(dataChunk->GetValue(column, row))));
}

template <>
Variant variantAt<TypeKind::TIMESTAMP>(
    ::duckdb::DataChunk* dataChunk,
    int32_t row,
    int32_t column) {
  return Variant::timestamp(duckdbTimestampToVelox(
      dataChunk->GetValue(column, row).GetValue<::duckdb::timestamp_t>()));
}

template <TypeKind kind>
Variant variantAt(const ::duckdb::Value& value) {
  if (value.type() == ::duckdb::LogicalType::INTERVAL) {
    return ::duckdb::Interval::GetMicro(value.GetValue<::duckdb::interval_t>());
  } else if (value.type() == ::duckdb::LogicalType::DATE) {
    return ::duckdb::Date::EpochDays(value.GetValue<::duckdb::date_t>());
  } else {
    // NOTE: duckdb only support native cpp type for GetValue so we need to use
    // DeepCopiedType instead of WrapperType here.
    using T = typename TypeTraits<kind>::DeepCopiedType;
    return Variant(value.GetValue<T>());
  }
}

template <>
Variant variantAt<TypeKind::HUGEINT>(const ::duckdb::Value& value) {
  auto hugeInt = ::duckdb::HugeIntValue::Get(value);
  return Variant(HugeInt::build(hugeInt.upper, hugeInt.lower));
}

template <>
Variant variantAt<TypeKind::TIMESTAMP>(const ::duckdb::Value& value) {
  return Variant::timestamp(
      duckdbTimestampToVelox(value.GetValue<::duckdb::timestamp_t>()));
}

template <>
Variant variantAt<TypeKind::VARCHAR>(const ::duckdb::Value& value) {
  return Variant(StringView(::duckdb::StringValue::Get(value)));
}

template <>
Variant variantAt<TypeKind::VARBINARY>(const ::duckdb::Value& value) {
  return Variant::binary(std::string(::duckdb::StringValue::Get(value)));
}

Variant nullVariant(const TypePtr& type) {
  return Variant(type->kind());
}

Variant arrayVariantAt(const ::duckdb::Value& vector, const TypePtr& arrayType);

Variant rowVariantAt(const ::duckdb::Value& vector, const TypePtr& rowType) {
  std::vector<Variant> values;
  const auto& structValue = ::duckdb::StructValue::GetChildren(vector);
  for (size_t i = 0; i < structValue.size(); ++i) {
    auto currChild = structValue[i];
    auto currType = rowType->childAt(i);
    if (currChild.IsNull()) {
      values.push_back(nullVariant(currType));
    } else if (currType->kind() == TypeKind::ROW) {
      values.push_back(rowVariantAt(currChild, currType));
    } else if (currType->kind() == TypeKind::ARRAY) {
      values.push_back(arrayVariantAt(currChild, currType));
    } else if (currType->isDecimal()) {
      values.push_back(duckdb::decimalVariant(currChild));
    } else {
      auto value = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          variantAt, currType->kind(), currChild);
      values.push_back(value);
    }
  }
  return Variant::row(std::move(values));
}

Variant mapVariantAt(const ::duckdb::Value& vector, const TypePtr& mapType) {
  std::map<Variant, Variant> map;

  auto mapTypePtr = dynamic_cast<const MapType*>(mapType.get());
  auto keyType = mapTypePtr->keyType();
  auto valueType = mapTypePtr->valueType();

  VELOX_CHECK(
      vector.type().id() == ::duckdb::LogicalTypeId::MAP,
      "Expected a MAP: {}",
      vector.type().ToString());

  const auto& valueList = ::duckdb::ListValue::GetChildren(vector);
  for (int i = 0; i < valueList.size(); i++) {
    Variant variantKey;
    Variant variantValue;
    auto value = ::duckdb::StructValue::GetChildren(valueList[i]);
    // Map key cannot be null.
    VELOX_CHECK(!value[0].IsNull());
    if (keyType->isDecimal()) {
      variantKey = duckdb::decimalVariant(value[0]);
    } else {
      variantKey = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          variantAt, keyType->kind(), value[0]);
    }

    if (value[1].IsNull()) {
      variantValue = nullVariant(valueType);
    } else if (valueType->isDecimal()) {
      variantValue = duckdb::decimalVariant(value[1]);
    } else {
      variantValue = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          variantAt, valueType->kind(), value[1]);
    }
    map.insert({variantKey, variantValue});
  }
  return Variant::map(map);
}

Variant arrayVariantAt(
    const ::duckdb::Value& vector,
    const TypePtr& arrayType) {
  std::vector<Variant> array;

  const auto& elementList = ::duckdb::ListValue::GetChildren(vector);

  auto arrayTypePtr = dynamic_cast<const ArrayType*>(arrayType.get());
  auto elementType = arrayTypePtr->elementType();
  for (int i = 0; i < elementList.size(); i++) {
    if (elementList[i].IsNull()) {
      array.push_back(nullVariant(elementType));
    } else if (elementType->kind() == TypeKind::ARRAY) {
      array.push_back(
          arrayVariantAt(elementList[i], arrayTypePtr->elementType()));
    } else if (elementType->kind() == TypeKind::ROW) {
      array.push_back(rowVariantAt(elementList[i], elementType));
    } else if (elementType->isDecimal()) {
      array.push_back(duckdb::decimalVariant(elementList[i]));
    } else {
      auto Variant = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          variantAt, elementType->kind(), elementList[i]);
      array.push_back(Variant);
    }
  }
  return Variant::array(std::move(array));
}

std::vector<MaterializedRow> materialize(
    ::duckdb::DataChunk* dataChunk,
    const RowTypePtr& rowType) {
  VELOX_CHECK_EQ(
      rowType->size(), dataChunk->GetTypes().size(), "Wrong number of columns");

  auto size = dataChunk->size();
  std::vector<MaterializedRow> rows;
  rows.reserve(size);

  // Pre-compute null values for all columns.
  std::vector<Variant> nulls;
  nulls.reserve(rowType->size());
  for (size_t i = 0; i < rowType->size(); ++i) {
    nulls.emplace_back(nullVariant(rowType->childAt(i)));
  }

  for (size_t i = 0; i < size; ++i) {
    MaterializedRow row;
    row.reserve(rowType->size());
    for (size_t j = 0; j < rowType->size(); ++j) {
      auto type = rowType->childAt(j);
      auto typeKind = type->kind();
      if (dataChunk->GetValue(j, i).IsNull()) {
        row.push_back(nulls[j]);
      } else if (typeKind == TypeKind::ARRAY) {
        row.push_back(arrayVariantAt(dataChunk->GetValue(j, i), type));
      } else if (typeKind == TypeKind::MAP) {
        row.push_back(mapVariantAt(dataChunk->GetValue(j, i), type));
      } else if (typeKind == TypeKind::ROW) {
        row.push_back(rowVariantAt(dataChunk->GetValue(j, i), type));
      } else if (type->isDecimal()) {
        row.push_back(duckdb::decimalVariant(dataChunk->GetValue(j, i)));
      } else if (type->isIntervalDayTime()) {
        auto value = Variant(
            ::duckdb::Interval::GetMicro(
                dataChunk->GetValue(j, i).GetValue<::duckdb::interval_t>()));
        row.push_back(value);
      } else if (type->isDate()) {
        auto value = Variant(
            ::duckdb::Date::EpochDays(
                dataChunk->GetValue(j, i).GetValue<::duckdb::date_t>()));
        row.push_back(value);
      } else if (type->isTime()) {
        VELOX_DCHECK(type->equivalent(*TIME()));
        // DuckDB TIME is in microseconds, Velox TIME is in milliseconds.
        auto value = Variant(
            dataChunk->GetValue(j, i).GetValue<::duckdb::dtime_t>().micros /
            1000L);
        row.push_back(value);
      } else {
        auto value = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            variantAt, typeKind, dataChunk, i, j);
        row.push_back(value);
      }
    }
    rows.push_back(row);
  }
  return rows;
}

void verifyDuckDBResult(
    const DuckDbQueryRunner::QueryResult& result,
    std::string_view sql) {
  VELOX_CHECK(
      !result->HasError(),
      "DuckDB query failed: {}\n{}",
      result->GetError(),
      sql);
}

} // namespace

::duckdb::DuckDB& DuckDbQueryRunner::database() {
  if (db_ == nullptr) {
    db_ = std::make_shared<::duckdb::DuckDB>(nullptr);
  }
  return *db_;
}

void DuckDbQueryRunner::createTable(
    const std::string& name,
    const std::vector<RowVectorPtr>& data) {
  auto query = fmt::format("DROP TABLE IF EXISTS {}", name);
  execute(query);

  auto& rowType = data[0]->type()->as<TypeKind::ROW>();
  ::duckdb::Connection con(database());
  auto sql = duckdb::makeCreateTableSql(name, rowType);
  auto res = con.Query(sql);
  verifyDuckDBResult(res, sql);

  ::duckdb::Appender appender(con, name);
  for (auto& vector : data) {
    // Resolve lazy vectors once per batch. duckValueAt casts each column to
    // SimpleVector, which returns null for a LazyVector, so materialize any
    // lazy child before the per-row loop.
    std::vector<VectorPtr> columnVectors;
    columnVectors.reserve(rowType.size());
    for (int32_t column = 0; column < rowType.size(); column++) {
      auto columnVector = vector->childAt(column);
      if (columnVector->isLazy()) {
        columnVector = BaseVector::loadedVectorShared(columnVector);
      }
      columnVectors.push_back(std::move(columnVector));
    }
    for (int32_t row = 0; row < vector->size(); row++) {
      appender.BeginRow();
      for (int32_t column = 0; column < rowType.size(); column++) {
        const auto& columnVector = columnVectors[column];
        auto type = rowType.childAt(column);
        if (columnVector->isNullAt(row)) {
          appender.Append(nullptr);
        } else if (type->isArray()) {
          appender.Append(duckValueAt<TypeKind::ARRAY>(columnVector, row));
        } else if (type->isMap()) {
          appender.Append(duckValueAt<TypeKind::MAP>(columnVector, row));
        } else if (type->isRow()) {
          appender.Append(duckValueAt<TypeKind::ROW>(columnVector, row));
        } else if (rowType.childAt(column)->isShortDecimal()) {
          appender.Append(duckValueAt<TypeKind::BIGINT>(columnVector, row));
        } else if (rowType.childAt(column)->isLongDecimal()) {
          appender.Append(duckValueAt<TypeKind::HUGEINT>(columnVector, row));
        } else {
          VELOX_CHECK(
              type->equivalent(*columnVector->type()),
              "{} vs. {}",
              type->toString(),
              columnVector->toString());
          auto value = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
              duckValueAt, type->kind(), columnVector, row);
          appender.Append(value);
        }
      }
      appender.EndRow();
    }
  }
}

DuckDbQueryRunner::QueryResult DuckDbQueryRunner::execute(
    const std::string& sql) {
  ::duckdb::Connection con(database());
  // Changing the default null order of NULLS FIRST used by DuckDB. Velox uses
  // NULLS LAST.
  con.Query("PRAGMA default_null_order='NULLS LAST'");
  auto duckDbResult = con.Query(sql);
  verifyDuckDBResult(duckDbResult, sql);
  return std::move(duckDbResult);
}

void DuckDbQueryRunner::execute(
    const std::string& sql,
    const RowTypePtr& resultRowType,
    std::function<void(std::vector<MaterializedRow>&)> resultCallback) {
  auto duckDbResult = execute(sql);

  for (;;) {
    auto dataChunk = duckDbResult->Fetch();
    verifyDuckDBResult(duckDbResult, sql);
    if (!dataChunk || (dataChunk->size() == 0)) {
      break;
    }
    auto rows = materialize(dataChunk.get(), resultRowType);
    resultCallback(rows);
  }
}

} // namespace facebook::velox::exec::test
