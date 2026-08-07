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

#include <string_view>

#include "velox/type/Type.h"
#include "velox/type/Variant.h"

namespace facebook::velox::common {
class Filter;
}

namespace facebook::velox::connector::hive {

/// kLocalTime shifts plain TIMESTAMP values from local time to UTC; kUtc does
/// not shift them. TIMESTAMP_UTC values are never shifted.
enum class TimestampMode {
  kLocalTime,
  kUtc,
};

/// kDateString parses an ISO date with DATE()->toDays(); kDaysSinceEpoch reads
/// an integer day count such as an Iceberg partition value.
enum class DateMode {
  kDateString,
  kDaysSinceEpoch,
};

/// Converts a non-null Hive partition string to a Variant of 'type' using the
/// supplied timestamp and date modes.
Variant partitionValueFromString(
    std::string_view value,
    const Type& type,
    TimestampMode timestampMode,
    DateMode dateMode);

/// Tests 'filter' against the value that partitionValueFromString() returns for
/// 'value'. Keeps partition pruning and partition reads on one conversion.
bool partitionValueMatchesFilter(
    std::string_view value,
    const Type& type,
    TimestampMode timestampMode,
    DateMode dateMode,
    const common::Filter& filter);

} // namespace facebook::velox::connector::hive
