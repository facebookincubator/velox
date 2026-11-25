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

#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {

/// Parses a trimmed Spark timestamp string, retaining any explicit timezone.
/// When allowed, time-only input uses today's date in the explicit timezone,
/// or in the session timezone if none is specified.
Expected<util::ParsedTimestampWithTimeZone> parseSparkTimestamp(
    const StringView& input,
    const tz::TimeZone* sessionTimeZone,
    bool allowTimeOnly);

/// Lenient conversion of local 'time' to the GMT time.
/// If the 'time' is invalid (DST gap), it will be corrected by
/// shifting forward to the nearest valid time.
void toGMTWithGapCorrection(Timestamp& time, const tz::TimeZone& zone);

} // namespace facebook::velox::functions::sparksql
