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

namespace facebook::velox::cudf_velox {

/// Registers GPU implementations of the Presto TIMESTAMP WITH TIME ZONE
/// function family: from_unixtime (with zone name or hour/minute offset),
/// to_unixtime, at_timezone, timezone_hour, timezone_minute, to_iso8601,
/// format_datetime, parse_datetime and from_iso8601_timestamp. date_format is
/// registered here too: it takes a plain TIMESTAMP rather than a zoned one, but
/// renders it in the session timezone through the same machinery
/// format_datetime uses. Names are prefixed with `prefix`.
///
/// now/current_timestamp are deliberately absent: they take no arguments, so
/// expression::optimize always constant folds them before an evaluator is
/// chosen, and a GPU implementation could never be reached.
void registerTimezoneFunctions(const std::string& prefix);

} // namespace facebook::velox::cudf_velox
