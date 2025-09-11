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
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/functions/prestosql/types/P4HyperLogLogType.h"

using namespace facebook::velox;

class P4HyperLogLogCastTest : public functions::test::CastBaseTest {};

TEST_F(P4HyperLogLogCastTest, nullValues) {
  auto nullData = std::vector<std::optional<StringView>>{
      std::nullopt, std::nullopt, std::nullopt, std::nullopt};

  // VARBINARY from/to P4HYPERLOGLOG
  testCast<StringView, StringView>(
      VARBINARY(), P4HYPERLOGLOG(), nullData, nullData);
  testCast<StringView, StringView>(
      P4HYPERLOGLOG(), VARBINARY(), nullData, nullData);

  // HYPERLOGLOG from/to P4HYPERLOGLOG
  testCast<StringView, StringView>(
      HYPERLOGLOG(), P4HYPERLOGLOG(), nullData, nullData);
  testCast<StringView, StringView>(
      P4HYPERLOGLOG(), HYPERLOGLOG(), nullData, nullData);
}

TEST_F(P4HyperLogLogCastTest, emptyValues) {
  auto emptyData =
      std::vector<std::optional<StringView>>{""_sv, ""_sv, ""_sv, ""_sv};

  // VARBINARY from/to P4HYPERLOGLOG
  testCast<StringView, StringView>(
      VARBINARY(), P4HYPERLOGLOG(), emptyData, emptyData);
  testCast<StringView, StringView>(
      P4HYPERLOGLOG(), VARBINARY(), emptyData, emptyData);

  // HYPERLOGLOG from/to P4HYPERLOGLOG
  testCast<StringView, StringView>(
      HYPERLOGLOG(), P4HYPERLOGLOG(), emptyData, emptyData);
  testCast<StringView, StringView>(
      P4HYPERLOGLOG(), HYPERLOGLOG(), emptyData, emptyData);
}

TEST_F(P4HyperLogLogCastTest, nonEmptyValues) {
  auto testData = std::vector<std::optional<StringView>>{
      "aaa"_sv, "test_hll_data"_sv, "xyz"_sv, std::nullopt};

  // VARBINARY from/to P4HYPERLOGLOG
  testCast<StringView, StringView>(
      VARBINARY(), P4HYPERLOGLOG(), testData, testData);
  testCast<StringView, StringView>(
      P4HYPERLOGLOG(), VARBINARY(), testData, testData);

  // HYPERLOGLOG from/to P4HYPERLOGLOG
  testCast<StringView, StringView>(
      HYPERLOGLOG(), P4HYPERLOGLOG(), testData, testData);
  testCast<StringView, StringView>(
      P4HYPERLOGLOG(), HYPERLOGLOG(), testData, testData);
}
