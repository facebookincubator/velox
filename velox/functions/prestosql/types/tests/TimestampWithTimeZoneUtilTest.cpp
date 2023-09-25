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

#include <random>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneUtil.h"

namespace facebook::velox::test {
TEST(TimestampWithTimeZoneUtilTest, pack) {
  std::mt19937 randGen(std::random_device{}());
  // 0xFFF8000000000000 and 0x7FFFFFFFFFFFF are hexadecimal numbers
  // that represent the minimum and maximum values that the
  // TimestampWithTimeZoneType type can represent, respectively.
  std::uniform_int_distribution<int64_t> millisUtcDis(
      0xFFF8000000000000L, 0x7FFFFFFFFFFFF);

  // 2233 represents the maximum value of timeZoneKey,
  // see tzDB in TimeZoneDatabase.cpp
  std::uniform_int_distribution<int16_t> timeZoneKeyDis(0, 2233);

  for (int64_t i = 0; i < 10'000; ++i) {
    auto millisUtc = millisUtcDis(randGen);
    auto timeZoneKey = timeZoneKeyDis(randGen);
    SCOPED_TRACE(
        fmt::format("millisUtc={}, timeZoneKey={}", millisUtc, timeZoneKey));

    auto packedTimeMillis = pack(millisUtc, timeZoneKey);
    ASSERT_EQ(unpackMillisUtc(packedTimeMillis), millisUtc);
    ASSERT_EQ(unpackZoneKeyId(packedTimeMillis), timeZoneKey);
  }
}
} // namespace facebook::velox::test
