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

#include "velox/functions/sparksql/specialforms/SparkTimeUtils.h"

#include <cstddef>
#include <cstdint>

#include "velox/type/Time.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::functions::sparksql {

Expected<int64_t> fromTimeStringMicros(const char* buf, size_t len) {
  auto componentsResult =
      util::parseTimeComponents(buf, len, /*requireSeconds*/ true, 6);
  if (componentsResult.hasError()) {
    return folly::makeUnexpected(componentsResult.error());
  }

  const auto components = componentsResult.value();

  if (components.hour < 0 || components.hour >= util::kHoursPerDay) {
    return folly::makeUnexpected(
        Status::UserError("Invalid hour value: {}", components.hour));
  }
  if (components.minute < 0 || components.minute >= util::kMinsPerHour) {
    return folly::makeUnexpected(
        Status::UserError("Invalid minute value: {}", components.minute));
  }
  if (components.second < 0 || components.second >= util::kSecsPerMinute) {
    return folly::makeUnexpected(
        Status::UserError("Invalid second value: {}", components.second));
  }
  if (components.fraction < 0 || components.fraction >= util::kMicrosPerSec) {
    return folly::makeUnexpected(
        Status::UserError(
            "Invalid microsecond value: {}", components.fraction));
  }

  constexpr int64_t kMicrosPerDay = util::kMicrosPerSec * util::kSecsPerDay;
  int64_t result =
      static_cast<int64_t>(components.hour) * util::kMicrosPerHour +
      static_cast<int64_t>(components.minute) * util::kMicrosPerMinute +
      static_cast<int64_t>(components.second) * util::kMicrosPerSec +
      static_cast<int64_t>(components.fraction);

  if (result < 0 || result >= kMicrosPerDay) {
    return folly::makeUnexpected(
        Status::UserError(
            "Time value {} is out of range [0, {})", result, kMicrosPerDay));
  }

  return result;
}

} // namespace facebook::velox::functions::sparksql
