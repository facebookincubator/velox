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

namespace facebook::velox::core {

/// Creates a constant vector of size 1 from a string representation of a value.
///
/// Converts string representations into constant vectors, supporting all scalar
/// types including primitives, dates, timestamps, decimals, and TIMESTAMP WITH
/// TIME ZONE. If @p value is nullopt, produces a null constant.
///
/// @param type Target Velox type.
/// @param value String value formatted like CAST(x AS VARCHAR). Date values use
/// ISO 8601 (YYYY-MM-DD). nullopt produces a null constant.
/// @param pool Memory pool for the result vector.
/// @param isLocalTimestamp When true, interprets TIMESTAMP strings as local
/// time and converts to GMT using the default session timezone.
/// @param isDaysSinceEpoch When true, parses DATE values as days since epoch
/// rather than ISO 8601 strings (used by Iceberg).
/// @param timezone When non-null, converts local TIMESTAMP values to GMT using
/// this zone (takes precedence over @p isLocalTimestamp).
VectorPtr newConstantFromString(
    const TypePtr& type,
    const std::optional<std::string>& value,
    memory::MemoryPool* pool,
    bool isLocalTimestamp = false,
    bool isDaysSinceEpoch = false,
    const tz::TimeZone* timezone = nullptr);

} // namespace facebook::velox::core
