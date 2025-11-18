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

#include <array>

#include "velox/type/Type.h"

namespace facebook::velox::fuzzer {

/// Input generator for TIME WITH TIME ZONE type.
/// Generates valid TIME WITH TIME ZONE values by combining:
/// - Time component: milliseconds since midnight (0 to 86399999)
/// - Timezone offset: bias-encoded minutes (-840 to 840 -> 0 to 1680)
/// Biases selection toward US timezone offsets (including DST) for better
/// coverage of DST-related edge cases.
class TimeWithTimezoneInputGenerator : public AbstractInputGenerator {
 public:
  // Frequently used timezone offsets in minutes (US timezones including DST)
  static constexpr std::array<int16_t, 7> kFrequentlyUsedOffsets = {
      -240, // EDT (Eastern Daylight Time)
      -300, // EST (Eastern Standard Time) / CDT (Central Daylight Time)
      -360, // CST (Central Standard Time) / MDT (Mountain Daylight Time)
      -420, // MST (Mountain Standard Time) / PDT (Pacific Daylight Time)
      -480, // PST (Pacific Standard Time) / AKDT (Alaska Daylight Time)
      -540, // AKST (Alaska Standard Time) / HDT (Hawaii-Aleutian Daylight Time)
      -600, // HST (Hawaii-Aleutian Standard Time)
  };

  TimeWithTimezoneInputGenerator(size_t seed, double nullRatio);

  Variant generate() override;
};

} // namespace facebook::velox::fuzzer
