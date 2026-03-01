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

#include "velox/functions/prestosql/tests/CastBaseTest.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/functions/sparksql/types/TimestampNTZRegistration.h"
#include "velox/functions/sparksql/types/TimestampNTZType.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class TimestampNTZCastTest : public functions::test::CastBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::sparksql::registerFunctions("");
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    registerTimestampNTZType();
  }
};

TEST_F(TimestampNTZCastTest, fromString) {
  // The timezone in the input string are allowed but ignored for TIMESTAMP_NTZ
  // values.
  std::vector<std::optional<std::string>> input{
      "1970-01-01",
      "1970-01-01 00:00:00-02:00",
      "1970-01-01 00:00:00 +02:00",
      "2000-01-01",
      "1970-01-01 00:00:00",
      "2000-01-01 12:21:56",
      "2015-03-18T12:03:17Z",
      "2015-03-18 12:03:17",
      "2015-03-18T12:03:17",
      "2015-03-18 12:03:17.123",
      "2015-03-18T12:03:17.123",
      "2015-03-18T12:03:17.456",
      "2015-03-18 12:03:17.456",
      "\n\f\r\t\n\u001F 2000-01-01 12:21:56\u000B\u001C\u001D\u001E",
  };
  std::vector<std::optional<TimestampNTZT::type>> expected{
      0,
      0,
      0,
      946684800000000,
      0,
      946729316000000,
      1426680197000000,
      1426680197000000,
      1426680197000000,
      1426680197123000,
      1426680197123000,
      1426680197456000,
      1426680197456000,
      946729316000000,
  };
  testCast<std::string, TimestampNTZT::type>(
      "timestamp_ntz", input, expected, VARCHAR(), TIMESTAMP_NTZ());

  // TimestampNTZ values should not be affected by timezone setting.
  setTimezone("Asia/Shanghai");
  testCast<std::string, TimestampNTZT::type>(
      "timestamp_ntz",
      {"1970-01-01 00:00:00",
       "1970-01-01 08:00:00",
       "1970-01-01 08:00:59",
       "1970"},
      {0, 8 * 3600 * 1000000L, 8 * 3600 * 1000000L + 59 * 1000000L, 0},
      VARCHAR(),
      TIMESTAMP_NTZ());
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
