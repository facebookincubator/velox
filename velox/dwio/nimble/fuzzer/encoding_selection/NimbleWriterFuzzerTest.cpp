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

// Thin gtest wrapper so CI can run the fuzzer loop. All setup and run logic is
// shared with the standalone binary via NimbleWriterFuzzerRunner.

#include <folly/init/Init.h>
#include <folly/json.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzer.h"
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzerRunner.h"

namespace facebook::nimble::fuzzer {

TEST(NimbleWriterFuzzerTest, cappingJsonShape) {
  bool sawMissing795{false};
  bool sawPresent795{false};
  bool sawMissing855{false};
  bool sawPresent855{false};
  bool sawEmptyT1{false};
  bool sawLongT1{false};
  bool sawPresent402{false};
  bool sawPresent921{false};

  for (uint64_t row = 0; row < 128; ++row) {
    const auto seed = 0xc0ffeeULL + row;
    const auto value = makeCappingJsonValue(seed, row);
    EXPECT_EQ(value, makeCappingJsonValue(seed, row));

    const auto parsed = folly::parseJson(value);
    ASSERT_TRUE(parsed.isObject());
    ASSERT_TRUE(parsed.count("t1"));
    const auto& t1 = parsed.at("t1");
    ASSERT_TRUE(t1.isArray());
    sawEmptyT1 |= t1.empty();
    sawLongT1 |= t1.size() >= 64;
    for (const auto& item : parsed.items()) {
      ASSERT_TRUE(item.second.isArray());
      for (const auto& element : item.second) {
        EXPECT_TRUE(element.isInt());
      }
    }
    sawMissing795 |= !parsed.count("795");
    sawPresent795 |= parsed.count("795");
    sawMissing855 |= !parsed.count("855");
    sawPresent855 |= parsed.count("855");
    sawPresent402 |= parsed.count("402");
    sawPresent921 |= parsed.count("921");
  }

  EXPECT_TRUE(sawMissing795);
  EXPECT_TRUE(sawPresent795);
  EXPECT_TRUE(sawMissing855);
  EXPECT_TRUE(sawPresent855);
  EXPECT_TRUE(sawEmptyT1);
  EXPECT_TRUE(sawLongT1);
  EXPECT_TRUE(sawPresent402);
  EXPECT_TRUE(sawPresent921);
}

TEST(NimbleWriterFuzzerTest, run) {
  runNimbleWriterFuzzer();
}

} // namespace facebook::nimble::fuzzer

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::nimble::fuzzer::setUpFuzzerEnvironments();
  return RUN_ALL_TESTS();
}
