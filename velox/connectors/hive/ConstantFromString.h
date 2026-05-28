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

#include <optional>
#include <string>

#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::tz {
class TimeZone;
}

namespace facebook::velox::connector::hive {

/// Creates a constant vector of size 1 from a string representation of a value.
///
/// Used to materialize partition column values and info columns (e.g., $path,
/// $file_size) when reading Hive, Iceberg and Delta Lake tables. Partition
/// values are stored as strings in HiveConnectorSplit::partitionKeys and need
/// to be converted to their appropriate types.
///
/// Parsing is shared with the partition filter path through
/// PartitionValue::fromString, so a partition value and a filter on it always
/// agree. TIMESTAMP WITH TIME ZONE is handled here rather than in
/// PartitionValue because its packed millis-plus-zone-key encoding is specific
/// to the vector representation.
///
/// @param type The target Velox type for the constant vector. Supports all
/// scalar types including primitives, dates, timestamps, decimals and
/// TIMESTAMP WITH TIME ZONE.
/// @param value The string representation of the value to convert, formatted
/// the same way as CAST(x as VARCHAR). Date values must be formatted using ISO
/// 8601 as YYYY-MM-DD. If nullopt, creates a null constant vector.
/// @param pool Memory pool for allocating the constant vector.
/// @param isLocalTimestamp If true and type is TIMESTAMP, interprets the string
/// value as local time and converts it to GMT. If false, treats the value as
/// already in GMT.
/// @param isDaysSinceEpoch If true and type is DATE, treats the string value as
/// an integer representing days since epoch (used by Iceberg). If false, parses
/// the string as a date string in ISO 8601 format (used by Hive).
/// @param timezone If non-null and type is TIMESTAMP, interprets the string
/// value as local time in this zone and converts it to GMT. Takes precedence
/// over 'isLocalTimestamp'.
///
/// @return A constant vector of size 1 containing the converted value, or a
/// null constant if value is nullopt.
/// @throws VeloxUserError if the string cannot be converted to the target type.
VectorPtr newConstantFromString(
    const TypePtr& type,
    const std::optional<std::string>& value,
    velox::memory::MemoryPool* pool,
    bool isLocalTimestamp,
    bool isDaysSinceEpoch,
    const tz::TimeZone* timezone = nullptr);

} // namespace facebook::velox::connector::hive
