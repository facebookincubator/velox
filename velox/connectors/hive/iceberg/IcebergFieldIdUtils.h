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
#pragma once

#include <string>
#include <unordered_map>

#include "velox/dwio/common/ParquetFieldId.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

/// Recursively extracts Iceberg field IDs from a ParquetFieldId tree into
/// a lowercase name -> fieldId mapping. Used during IcebergDataSource
/// construction to build the map passed to the Parquet reader for field-ID-
/// based column matching.
void extractNestedFieldIds(
    const parquet::ParquetFieldId& field,
    const RowTypePtr& rowType,
    std::unordered_map<std::string, int32_t>& mapping);

} // namespace facebook::velox::connector::hive::iceberg
