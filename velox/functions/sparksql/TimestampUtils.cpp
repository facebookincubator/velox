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
#include "velox/functions/sparksql/TimestampUtils.h"

#include <algorithm>

namespace facebook::velox::functions::sparksql {

Expected<util::ParsedTimestampWithTimeZone> parseSparkTimestamp(
    const StringView& input,
    const tz::TimeZone* sessionTimeZone,
    bool allowTimeOnly) {
  std::string_view str(input.data(), input.size());
  const auto colon = str.find(':');
  const bool timeOnly = allowTimeOnly && !str.empty() &&
      (str.front() == 'T' ||
       (colon >= 1 && colon <= 2 &&
        std::all_of(str.begin(), str.begin() + colon, [](char c) {
          return c >= '0' && c <= '9';
        })));
  std::string datedInput;
  if (timeOnly) {
    // Reuse the timestamp parser, then select today's date after
    // resolving the explicit or session timezone.
    datedInput = "1970-01-01T";
    datedInput.append(str.front() == 'T' ? str.substr(1) : str);
    str = datedInput;
  }
  auto parsed = util::fromTimestampWithTimezoneString(
      str.data(), str.size(), util::TimestampParseMode::kSparkCast);
  if (timeOnly && parsed.hasValue()) {
    auto& value = parsed.value();
    const auto* zone = value.timeZone ? value.timeZone : sessionTimeZone;
    const auto days = util::toDate(Timestamp::now(), zone);
    value.timestamp = Timestamp(
        static_cast<int64_t>(days) * util::kSecsPerDay +
            value.timestamp.getSeconds(),
        value.timestamp.getNanos());
  }
  return parsed;
}

void toGMTWithGapCorrection(Timestamp& time, const tz::TimeZone& zone) {
  time = Timestamp(
      zone.correct_nonexistent_time(std::chrono::seconds(time.getSeconds()))
          .count(),
      time.getNanos());
  time.toGMT(zone);
}

} // namespace facebook::velox::functions::sparksql
