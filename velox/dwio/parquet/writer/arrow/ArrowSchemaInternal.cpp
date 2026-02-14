/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// Adapted from Apache Arrow.

#include "velox/dwio/parquet/writer/arrow/ArrowSchemaInternal.h"

#include "arrow/type.h"

using ArrowType = ::arrow::DataType;
using ArrowTypeId = ::arrow::Type;
using ParquetType = ::facebook::velox::parquet::arrow::Type;

namespace facebook::velox::parquet::arrow::arrow {

using ::arrow::Result;
using ::arrow::Status;
using ::arrow::internal::checked_cast;

Result<std::shared_ptr<ArrowType>> makeArrowDecimal(
    const LogicalType& logicalType) {
  const auto& decimal = checked_cast<const DecimalLogicalType&>(logicalType);
  if (decimal.precision() <= ::arrow::Decimal128Type::kMaxPrecision) {
    return ::arrow::Decimal128Type::Make(decimal.precision(), decimal.scale());
  }
  return ::arrow::Decimal256Type::Make(decimal.precision(), decimal.scale());
}

Result<std::shared_ptr<ArrowType>> makeArrowInt(
    const LogicalType& logicalType) {
  const auto& integer = checked_cast<const IntLogicalType&>(logicalType);
  switch (integer.bitWidth()) {
    case 8:
      return integer.isSigned() ? ::arrow::int8() : ::arrow::uint8();
    case 16:
      return integer.isSigned() ? ::arrow::int16() : ::arrow::uint16();
    case 32:
      return integer.isSigned() ? ::arrow::int32() : ::arrow::uint32();
    default:
      return Status::TypeError(
          logicalType.toString(), " can not annotate physical type Int32");
  }
}

Result<std::shared_ptr<ArrowType>> makeArrowInt64(
    const LogicalType& logicalType) {
  const auto& integer = checked_cast<const IntLogicalType&>(logicalType);
  switch (integer.bitWidth()) {
    case 64:
      return integer.isSigned() ? ::arrow::int64() : ::arrow::uint64();
    default:
      return Status::TypeError(
          logicalType.toString(), " can not annotate physical type Int64");
  }
}

Result<std::shared_ptr<ArrowType>> makeArrowTime32(
    const LogicalType& logicalType) {
  const auto& time = checked_cast<const TimeLogicalType&>(logicalType);
  switch (time.timeUnit()) {
    case LogicalType::TimeUnit::kMillis:
      return ::arrow::time32(::arrow::TimeUnit::MILLI);
    default:
      return Status::TypeError(
          logicalType.toString(), " can not annotate physical type Time32");
  }
}

Result<std::shared_ptr<ArrowType>> makeArrowTime64(
    const LogicalType& logicalType) {
  const auto& time = checked_cast<const TimeLogicalType&>(logicalType);
  switch (time.timeUnit()) {
    case LogicalType::TimeUnit::kMicros:
      return ::arrow::time64(::arrow::TimeUnit::MICRO);
    case LogicalType::TimeUnit::kNanos:
      return ::arrow::time64(::arrow::TimeUnit::NANO);
    default:
      return Status::TypeError(
          logicalType.toString(), " can not annotate physical type Time64");
  }
}

Result<std::shared_ptr<ArrowType>> makeArrowTimestamp(
    const LogicalType& logicalType) {
  const auto& timestamp =
      checked_cast<const TimestampLogicalType&>(logicalType);
  const bool utcNormalized =
      timestamp.isFromConvertedType() ? false : timestamp.isAdjustedToUtc();
  static const char* utcTimezone = "UTC";
  switch (timestamp.timeUnit()) {
    case LogicalType::TimeUnit::kMillis:
      return (
          utcNormalized
              ? ::arrow::timestamp(::arrow::TimeUnit::MILLI, utcTimezone)
              : ::arrow::timestamp(::arrow::TimeUnit::MILLI));
    case LogicalType::TimeUnit::kMicros:
      return (
          utcNormalized
              ? ::arrow::timestamp(::arrow::TimeUnit::MICRO, utcTimezone)
              : ::arrow::timestamp(::arrow::TimeUnit::MICRO));
    case LogicalType::TimeUnit::kNanos:
      return (
          utcNormalized
              ? ::arrow::timestamp(::arrow::TimeUnit::NANO, utcTimezone)
              : ::arrow::timestamp(::arrow::TimeUnit::NANO));
    default:
      return Status::TypeError(
          "Unrecognized time unit in timestamp logical_type: ",
          logicalType.toString());
  }
}

Result<std::shared_ptr<ArrowType>> fromByteArray(
    const LogicalType& logicalType) {
  switch (logicalType.type()) {
    case LogicalType::Type::kString:
      return ::arrow::utf8();
    case LogicalType::Type::kDecimal:
      return makeArrowDecimal(logicalType);
    case LogicalType::Type::kNone:
    case LogicalType::Type::kEnum:
    case LogicalType::Type::kJson:
    case LogicalType::Type::kBson:
      return ::arrow::binary();
    default:
      return Status::NotImplemented(
          "Unhandled logical logical_type ",
          logicalType.toString(),
          " for binary array");
  }
}

Result<std::shared_ptr<ArrowType>> fromFLBA(
    const LogicalType& logicalType,
    int32_t physicalLength) {
  switch (logicalType.type()) {
    case LogicalType::Type::kDecimal:
      return makeArrowDecimal(logicalType);
    case LogicalType::Type::kNone:
    case LogicalType::Type::kInterval:
    case LogicalType::Type::kUuid:
      return ::arrow::fixed_size_binary(physicalLength);
    default:
      return Status::NotImplemented(
          "Unhandled logical logical_type ",
          logicalType.toString(),
          " for fixed-length binary array");
  }
}

::arrow::Result<std::shared_ptr<ArrowType>> fromInt32(
    const LogicalType& logicalType) {
  switch (logicalType.type()) {
    case LogicalType::Type::kInt:
      return makeArrowInt(logicalType);
    case LogicalType::Type::kDate:
      return ::arrow::date32();
    case LogicalType::Type::kTime:
      return makeArrowTime32(logicalType);
    case LogicalType::Type::kDecimal:
      return makeArrowDecimal(logicalType);
    case LogicalType::Type::kNone:
      return ::arrow::int32();
    default:
      return Status::NotImplemented(
          "Unhandled logical type ", logicalType.toString(), " for INT32");
  }
}

Result<std::shared_ptr<ArrowType>> fromInt64(const LogicalType& logicalType) {
  switch (logicalType.type()) {
    case LogicalType::Type::kInt:
      return makeArrowInt64(logicalType);
    case LogicalType::Type::kDecimal:
      return makeArrowDecimal(logicalType);
    case LogicalType::Type::kTimestamp:
      return makeArrowTimestamp(logicalType);
    case LogicalType::Type::kTime:
      return makeArrowTime64(logicalType);
    case LogicalType::Type::kNone:
      return ::arrow::int64();
    default:
      return Status::NotImplemented(
          "Unhandled logical type ", logicalType.toString(), " for INT64");
  }
}

Result<std::shared_ptr<ArrowType>> getArrowType(
    Type::type physicalType,
    const LogicalType& logicalType,
    int typeLength,
    const ::arrow::TimeUnit::type int96ArrowTimeUnit) {
  if (logicalType.isInvalid() || logicalType.isNull()) {
    return ::arrow::null();
  }

  switch (physicalType) {
    case ParquetType::kBoolean:
      return ::arrow::boolean();
    case ParquetType::kInt32:
      return fromInt32(logicalType);
    case ParquetType::kInt64:
      return fromInt64(logicalType);
    case ParquetType::kInt96:
      return ::arrow::timestamp(int96ArrowTimeUnit);
    case ParquetType::kFloat:
      return ::arrow::float32();
    case ParquetType::kDouble:
      return ::arrow::float64();
    case ParquetType::kByteArray:
      return fromByteArray(logicalType);
    case ParquetType::kFixedLenByteArray:
      return fromFLBA(logicalType, typeLength);
    default: {
      // PARQUET-1565: This can occur if the file is corrupt.
      return Status::IOError(
          "Invalid physical column type: ", typeToString(physicalType));
    }
  }
}

Result<std::shared_ptr<ArrowType>> getArrowType(
    const schema::PrimitiveNode& primitive,
    const ::arrow::TimeUnit::type int96ArrowTimeUnit) {
  return getArrowType(
      primitive.physicalType(),
      *primitive.logicalType(),
      primitive.typeLength(),
      int96ArrowTimeUnit);
}

} // namespace facebook::velox::parquet::arrow::arrow
