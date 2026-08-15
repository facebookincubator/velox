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

// Adapted from Apache Arrow.

#pragma once

#include "arrow/result.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"

namespace arrow {
class DataType;
}

namespace facebook::velox::parquet::arrow::arrow {

using ::arrow::Result;

Result<std::shared_ptr<::arrow::DataType>> fromByteArray(
    const LogicalType& logicalType);
Result<std::shared_ptr<::arrow::DataType>> fromFLBA(
    const LogicalType& logicalType,
    int32_t physicalLength);
Result<std::shared_ptr<::arrow::DataType>> fromInt32(
    const LogicalType& logicalType);
Result<std::shared_ptr<::arrow::DataType>> fromInt64(
    const LogicalType& logicalType);

Result<std::shared_ptr<::arrow::DataType>> getArrowType(
    Type::type physicalType,
    const LogicalType& logicalType,
    int typeLength);

Result<std::shared_ptr<::arrow::DataType>> getArrowType(
    Type::type physicalType,
    const LogicalType& logicalType,
    int typeLength,
    ::arrow::TimeUnit::type int96ArrowTimeUnit = ::arrow::TimeUnit::NANO);

Result<std::shared_ptr<::arrow::DataType>> getArrowType(
    const schema::PrimitiveNode& primitive,
    ::arrow::TimeUnit::type int96ArrowTimeUnit = ::arrow::TimeUnit::NANO);

} // namespace facebook::velox::parquet::arrow::arrow
