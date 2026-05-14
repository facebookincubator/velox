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

#include "velox/functions/prestosql/types/fuzzer_utils/TimestampWithTimeZoneInputGenerator.h"

#include <gtest/gtest.h>

#include "velox/functions/lib/DateTimeFormatterBuilder.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Variant.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::fuzzer::test {
namespace {

bool roundTripsThroughPrestoQueryRunnerTransform(
    int64_t timestampWithTimeZone) {
  static const auto formatter =
      functions::buildJodaDateTimeFormatter("yyyy-MM-dd HH:mm:ss.SSS ZZZ")
          .value();

  const auto input =
      TIMESTAMP_WITH_TIME_ZONE()->valueToString(timestampWithTimeZone);
  auto parsed = formatter->parse(input);
  if (parsed.hasError() || parsed->timezone == nullptr) {
    return false;
  }

  auto timestamp = parsed->timestamp;
  timestamp.toGMT(*parsed->timezone);
  return pack(timestamp, parsed->timezone->id()) == timestampWithTimeZone;
}

} // namespace

TEST(TimestampWithTimeZoneInputGeneratorTest, generate) {
  TimestampWithTimeZoneInputGenerator generator(123456, 0.1);

  size_t numTrials = 100;
  for (size_t i = 0; i < numTrials; ++i) {
    variant generated = generator.generate();

    if (generated.isNull()) {
      continue;
    }

    generated.checkIsKind(TypeKind::BIGINT);
    const auto value = generated.value<TypeKind::BIGINT>();

    // The value can be any random int64_t with the one restriction that the
    // time zone should be valid.
    auto zoneKey = unpackZoneKeyId(value);
    EXPECT_NE(tz::locateZone(zoneKey), nullptr);
    EXPECT_TRUE(roundTripsThroughPrestoQueryRunnerTransform(value));
  }
}

TEST(TimestampWithTimeZoneInputGeneratorTest, skipsAmbiguousRoundTrips) {
  const auto* timeZone = tz::locateZone("America/Los_Angeles");
  ASSERT_NE(timeZone, nullptr);

  // 2019-11-03 09:00:00 UTC is 2019-11-03 01:00:00 local after DST falls
  // back. Formatting with a zone id loses which occurrence of 01:00 this was,
  // so parse_datetime(...) resolves it to the earlier instant.
  const auto ambiguousValue =
      pack(Timestamp::fromMillis(1572771600000), timeZone->id());

  EXPECT_FALSE(roundTripsThroughPrestoQueryRunnerTransform(ambiguousValue));
}
} // namespace facebook::velox::fuzzer::test
