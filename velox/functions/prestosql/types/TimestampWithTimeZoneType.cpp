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

#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/functions/lib/DateTimeFormatter.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox {

std::string TimestampWithTimeZoneType::valueToString(int64_t value) const {
  static auto kFormatter =
      functions::buildJodaDateTimeFormatter("yyyy-MM-dd HH:mm:ss.SSS ZZZ")
          .value();

  auto timestamp = unpackTimestampUtc(value);
  auto* timeZone = tz::locateZone(tz::getTimeZoneName(unpackZoneKeyId(value)));

  auto maxSize = kFormatter->maxResultSize(timeZone);
  std::string result(maxSize, '\0');
  auto size = kFormatter->format(timestamp, timeZone, maxSize, result.data());
  result.resize(size);
  return result;
}

} // namespace facebook::velox
