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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergFilterTransform.h"

#include "velox/type/Filter.h"
#include "velox/type/Type.h"

#include <optional>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Returns whether a DATE partition value is encoded as the number of days
/// since the epoch.
///
/// The two encodings are format-disjoint: Iceberg writes days since the epoch
/// (e.g. "20244"), while Hive-migrated tables write a date string (e.g.
/// "2025-06-05") whose separators fail an integer parse.
///
/// Folding a filter and materializing the column must agree on the encoding,
/// so both go through here.
bool isDaysSinceEpoch(
    const TypePtr& type,
    const std::optional<std::string>& value);

/// Evaluates a filter over a whole column against the constant value that
/// column holds, for a split where the column is not read from the data file.
///
/// @param filter Filter over the whole column.
/// @param type Type of the column.
/// @param value The column's value for the whole split, `nullopt` for a column
/// missing from the data file, which is NULL throughout.
/// @param readTimestampAsLocalTime Whether a TIMESTAMP value is a local time
/// to shift to UTC rather than a UTC time.
/// @return Whether the filter accepts or rejects the split.
ConstantFilterFold foldFilterOnConstant(
    const common::Filter& filter,
    const TypePtr& type,
    const std::optional<std::string>& value,
    bool readTimestampAsLocalTime);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
