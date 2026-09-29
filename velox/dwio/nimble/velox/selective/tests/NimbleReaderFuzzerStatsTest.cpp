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

#include "velox/common/base/ConcurrentRuntimeStatWriter.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/dwio/nimble/velox/selective/NimbleReaderFuzzerStats.h"

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

  updateStringDictionaryEncodingPreserved();
  updateStringDictionaryEncodingAbandoned();

  const auto stats = writer.runtimeStats();
#ifdef NIMBLE_READER_FUZZER_STATS_ENABLED
  EXPECT_EQ(stats.at(std::string(kStringDictionaryEncodingPreserved)).sum, 1);
  EXPECT_EQ(stats.at(std::string(kStringDictionaryEncodingAbandoned)).sum, 1);
#else
  EXPECT_THAT(stats, testing::IsEmpty());
#endif
}

} // namespace facebook::nimble::fuzzer::test
