/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <gtest/gtest.h>
#include <unordered_set>

#include "velox/common/fuzzer/Utils.h"
#include "velox/type/Time.h"
#include "velox/type/Variant.h"

namespace facebook::velox::fuzzer::test {

TEST(TimeWithTimezoneInputGeneratorTest, generate) {
  TimeWithTimezoneInputGenerator generator(123456, 0.1);

  size_t numTrials = 100;
  for (size_t i = 0; i < numTrials; ++i) {
    variant generated = generator.generate();

    if (generated.isNull()) {
      continue;
    }

    generated.checkIsKind(TypeKind::BIGINT);
    const auto value = generated.value<TypeKind::BIGINT>();

    // Verify time component is in valid range [0, kMillisInDay)
    const auto timeMillis = util::unpackMillisUtc(value);
    EXPECT_GE(timeMillis, 0);
    EXPECT_LT(timeMillis, util::kMillisInDay);

    // Verify timezone offset is in valid bias-encoded range [0, 1680]
    const auto timezoneOffset = util::unpackZoneOffset(value);
    EXPECT_GE(timezoneOffset, 0);
    EXPECT_LE(timezoneOffset, 2 * util::kTimeZoneBias);
  }
}

TEST(TimeWithTimezoneInputGeneratorTest, generatesBothOffsetTypes) {
  TimeWithTimezoneInputGenerator generator(12345, 0.0); // Fixed seed, no nulls

  std::unordered_set<int16_t> frequentlyUsed(
      kFrequentlyUsedTimezoneOffsets.begin(),
      kFrequentlyUsedTimezoneOffsets.end());

  bool foundFrequentlyUsed = false;
  bool foundOther = false;

  for (int i = 0; i < 100 && (!foundFrequentlyUsed || !foundOther); ++i) {
    auto value = generator.generate().value<TypeKind::BIGINT>();
    auto encoded = util::unpackZoneOffset(value);
    auto offset = util::decodeTimezoneOffset(encoded);

    if (frequentlyUsed.count(offset)) {
      foundFrequentlyUsed = true;
    } else {
      foundOther = true;
    }
  }

  EXPECT_TRUE(foundFrequentlyUsed);
  EXPECT_TRUE(foundOther);
}

} // namespace facebook::velox::fuzzer::test
