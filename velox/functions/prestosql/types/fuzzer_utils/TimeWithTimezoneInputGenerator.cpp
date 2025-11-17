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

#include "velox/functions/prestosql/types/fuzzer_utils/TimeWithTimezoneInputGenerator.h"

#include "velox/common/fuzzer/Utils.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/type/Time.h"
#include "velox/type/Variant.h"

namespace facebook::velox::fuzzer {

TimeWithTimezoneInputGenerator::TimeWithTimezoneInputGenerator(
    size_t seed,
    double nullRatio)
    : AbstractInputGenerator(
          seed,
          TIME_WITH_TIME_ZONE(),
          nullptr /*leafGenerator*/,
          nullRatio) {}

variant TimeWithTimezoneInputGenerator::generate() {
  if (coinToss(rng_, nullRatio_)) {
    return variant::null(type_->kind());
  }

  const int64_t timeMillis = rand<int64_t>(rng_, 0, util::kMillisInDay - 1);

  int16_t timezoneOffsetMinutes;
  // 25% of the time, pick from frequently used timezone offsets
  if (coinToss(rng_, 0.25)) {
    timezoneOffsetMinutes = kFrequentlyUsedOffsets[rand<size_t>(
        rng_, 0, kFrequentlyUsedOffsets.size() - 1)];
  } else {
    // rest, pick a random timezone offset for variety
    timezoneOffsetMinutes =
        rand<int16_t>(rng_, -util::kTimeZoneBias, util::kTimeZoneBias);
  }

  const int16_t biasEncodedTimezone = util::biasEncode(timezoneOffsetMinutes);
  return util::pack(timeMillis, biasEncodedTimezone);
}

} // namespace facebook::velox::fuzzer
