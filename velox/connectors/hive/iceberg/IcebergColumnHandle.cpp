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

#include <string>
#include <utility>
#include <vector>

#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/dwio/parquet/ParquetFieldId.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

IcebergColumnHandle::IcebergColumnHandle(
    const std::string& name,
    ColumnType columnType,
    TypePtr dataType,
    parquet::ParquetFieldId icebergField,
    std::vector<common::Subfield> requiredSubfields)
    : HiveColumnHandle(
          name,
          columnType,
          dataType,
          dataType,
          std::move(requiredSubfields),
          ColumnParseParameters{ColumnParseParameters::
                                    PartitionDateValueFormat::kDaysSinceEpoch}),
      field_(std::move(icebergField)) {}

const parquet::ParquetFieldId& IcebergColumnHandle::field() const {
  return field_;
}

} // namespace facebook::velox::connector::hive::iceberg
