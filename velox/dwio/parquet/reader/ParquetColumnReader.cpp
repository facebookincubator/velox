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

//
// Created by Ying Su on 2/14/22.
//

#include "velox/dwio/parquet/reader/ParquetColumnReader.h"
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"
#include "velox/dwio/parquet/reader/BooleanColumnReader.h"
#include "velox/dwio/parquet/reader/FloatingPointColumnReader.h"
#include "velox/dwio/parquet/reader/IntegerColumnReader.h"
#include "velox/dwio/parquet/reader/RepeatedColumnReader.h"
#include "velox/dwio/parquet/reader/StringColumnReader.h"
#include "velox/dwio/parquet/reader/StructColumnReader.h"

#include "velox/dwio/parquet/reader/Statistics.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

// static
std::unique_ptr<dwio::common::SelectiveColumnReader> ParquetColumnReader::build(
    const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
    ParquetParams& params,
    common::ScanSpec& scanSpec) {
  auto colName = scanSpec.fieldName();

  switch (dataType->type->kind()) {
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT:
    case TypeKind::DATE:
    case TypeKind::SHORT_DECIMAL:
    case TypeKind::LONG_DECIMAL:
      return std::make_unique<IntegerColumnReader>(
          dataType, dataType, params, scanSpec);

    case TypeKind::REAL:
      return std::make_unique<FloatingPointColumnReader<float, float>>(
          dataType, dataType->type, params, scanSpec);
    case TypeKind::DOUBLE:
      return std::make_unique<FloatingPointColumnReader<double, double>>(
          dataType, dataType->type, params, scanSpec);

    case TypeKind::ROW:
      return std::make_unique<StructColumnReader>(dataType, params, scanSpec);

    case TypeKind::VARBINARY:
    case TypeKind::VARCHAR:
      return std::make_unique<StringColumnReader>(dataType, params, scanSpec);

    case TypeKind::ARRAY:
      return std::make_unique<ListColumnReader>(dataType, params, scanSpec);

    case TypeKind::MAP:
      return std::make_unique<MapColumnReader>(dataType, params, scanSpec);

    case TypeKind::BOOLEAN:
      return std::make_unique<BooleanColumnReader>(dataType, params, scanSpec);

    default:
      VELOX_FAIL(
          "buildReader unhandled type: " +
          mapTypeKindToName(dataType->type->kind()));
  }
}

} // namespace facebook::velox::parquet
