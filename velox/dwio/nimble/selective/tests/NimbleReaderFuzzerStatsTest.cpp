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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <map>

#include "velox/common/base/ConcurrentRuntimeStatWriter.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/dwio/nimble/selective/NimbleReaderFuzzerStats.h"
#include "velox/type/Filter.h"

namespace facebook::nimble::fuzzer::test {

TEST(NimbleReaderFuzzerStatsTest, publishedNames) {
  EXPECT_EQ(
      kStringDictionaryEncodingPreserved,
      "nimbleStringDictionaryEncodingPreserved");
  EXPECT_EQ(
      kStringDictionaryEncodingAbandoned,
      "nimbleStringDictionaryEncodingAbandoned");
}

TEST(NimbleReaderFuzzerStatsTest, updates) {
  velox::ConcurrentRuntimeStatWriter writer;
  velox::RuntimeStatWriterScopeGuard guard(&writer);

  const velox::common::AlwaysTrue noFilter;
  updateStringDictionaryEncodingPreserved();
  updateStringDictionaryEncodingAbandoned();
  updateSmallintDictionaryAwareReads(noFilter);
  updateIntegerDictionaryAwareReads(noFilter);
  updateIntegerDictionaryAwareReads(noFilter);
  updateBigintDictionaryAwareReads(noFilter);
  updateRealDictionaryAwareReads(noFilter);
  updateDoubleDictionaryAwareReads(noFilter);

  const auto stats = writer.runtimeStats();
#ifdef NIMBLE_READER_FUZZER_STATS_ENABLED
  EXPECT_EQ(stats.at(std::string(kStringDictionaryEncodingPreserved)).sum, 1);
  EXPECT_EQ(stats.at(std::string(kStringDictionaryEncodingAbandoned)).sum, 1);
  EXPECT_EQ(stats.at(std::string(kSmallintDictionaryAwareReads)).sum, 1);
  EXPECT_EQ(stats.at(std::string(kIntegerDictionaryAwareReads)).sum, 2);
  EXPECT_EQ(stats.at(std::string(kBigintDictionaryAwareReads)).sum, 1);
  EXPECT_EQ(stats.at(std::string(kRealDictionaryAwareReads)).sum, 1);
  EXPECT_EQ(stats.at(std::string(kDoubleDictionaryAwareReads)).sum, 1);
#else
  EXPECT_THAT(stats, testing::IsEmpty());
#endif
}

TEST(NimbleReaderFuzzerStatsTest, dictionaryAwareReadsByFilterKind) {
  EXPECT_EQ(
      dictionaryAwareReadsByFilterKind(
          kIntegerDictionaryAwareReads, "BigintRange"),
      "nimbleIntegerDictionaryAwareReads.BigintRange");

  velox::ConcurrentRuntimeStatWriter writer;
  velox::RuntimeStatWriterScopeGuard guard(&writer);

  updateIntegerDictionaryAwareReads(velox::common::AlwaysTrue());
  updateIntegerDictionaryAwareReads(
      velox::common::BigintRange(0, 10, /*nullAllowed=*/false));
  updateIntegerDictionaryAwareReads(
      velox::common::BigintRange(5, 20, /*nullAllowed=*/true));
  updateDoubleDictionaryAwareReads(velox::common::IsNotNull());

  std::map<std::string, int64_t> sums;
  for (const auto& [name, metric] : writer.runtimeStats()) {
    sums.emplace(name, metric.sum);
  }
#ifdef NIMBLE_READER_FUZZER_STATS_ENABLED
  EXPECT_THAT(
      sums,
      testing::ElementsAre(
          testing::Pair("nimbleDoubleDictionaryAwareReads", 1),
          testing::Pair("nimbleDoubleDictionaryAwareReads.IsNotNull", 1),
          testing::Pair("nimbleIntegerDictionaryAwareReads", 3),
          testing::Pair("nimbleIntegerDictionaryAwareReads.AlwaysTrue", 1),
          testing::Pair("nimbleIntegerDictionaryAwareReads.BigintRange", 2)));
#else
  EXPECT_THAT(sums, testing::IsEmpty());
#endif
}

} // namespace facebook::nimble::fuzzer::test
