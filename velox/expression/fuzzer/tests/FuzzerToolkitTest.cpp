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

#include "velox/expression/fuzzer/FuzzerToolkit.h"

#include "velox/exec/Aggregate.h"

#include <gtest/gtest.h>
#include "velox/common/testutil/TempFilePath.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::fuzzer::test {
using namespace facebook::velox::common::testutil;

class FuzzerToolKitTest : public testing::Test,
                          public facebook::velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  bool compareBuffers(const BufferPtr& lhs, const BufferPtr& rhs) {
    if (!lhs && !rhs) {
      return true;
    }
    if ((lhs && !rhs) || (!lhs && rhs) || lhs->size() != rhs->size()) {
      return false;
    }
    return memcmp(lhs->as<char>(), rhs->as<char>(), lhs->size()) == 0;
  }

  bool equals(const InputRowMetadata& lhs, const InputRowMetadata& rhs) {
    return lhs.columnsToWrapInLazy == rhs.columnsToWrapInLazy &&
        lhs.columnsToWrapInCommonDictionary ==
        rhs.columnsToWrapInCommonDictionary;
  }
};

TEST_F(FuzzerToolKitTest, inputRowMetadataRoundTrip) {
  InputRowMetadata metadata;
  metadata.columnsToWrapInLazy = {1, -2, 3, -4, 5};
  metadata.columnsToWrapInCommonDictionary = {1, 2, 3, 4, 5};

  {
    auto path = TempFilePath::create();
    metadata.saveToFile(path->getPath().c_str());
    auto copy =
        InputRowMetadata::restoreFromFile(path->getPath().c_str(), pool());
    ASSERT_TRUE(equals(metadata, copy));
  }
}
TEST_F(FuzzerToolKitTest, parseFunctionNames) {
  EXPECT_TRUE(parseFunctionNames("").empty());
  EXPECT_TRUE(parseFunctionNames(" , ").empty());
  EXPECT_EQ(
      parseFunctionNames(" Min, MAX ,, sum"),
      (std::unordered_set<std::string>{"min", "max", "sum"}));
}

TEST_F(FuzzerToolKitTest, onlyContainsSkippedFunctions) {
  const std::unordered_set<std::string> skip{"bloom_filter_agg", "Min"};

  // An empty 'only' list asks for every function, so it is never fully
  // skipped.
  EXPECT_FALSE(onlyContainsSkippedFunctions("", skip));
  EXPECT_FALSE(onlyContainsSkippedFunctions("", {}));
  EXPECT_FALSE(onlyContainsSkippedFunctions(" , ", skip));

  // Names are matched after trimming and lower casing, on both lists.
  EXPECT_TRUE(onlyContainsSkippedFunctions("bloom_filter_agg", skip));
  EXPECT_TRUE(onlyContainsSkippedFunctions(" BLOOM_FILTER_AGG , min ", skip));
  EXPECT_TRUE(onlyContainsSkippedFunctions("min,bloom_filter_agg", skip));

  // One function that is not skipped is enough to run the fuzzer.
  EXPECT_FALSE(onlyContainsSkippedFunctions("min,sum", skip));
  EXPECT_FALSE(onlyContainsSkippedFunctions("sum", skip));
  EXPECT_FALSE(onlyContainsSkippedFunctions("min", {}));
}

TEST_F(FuzzerToolKitTest, filterSignatures) {
  exec::AggregateFunctionSignatureMap input;
  for (const auto& name : {"min", "max", "sum"}) {
    input[name] = {};
  }

  EXPECT_EQ(filterSignatures(input, "", {}).size(), 3);
  EXPECT_EQ(filterSignatures(input, "min, MAX", {}).size(), 2);
  EXPECT_EQ(filterSignatures(input, "", {"Min"}).size(), 2);

  // A skipped function is dropped even when 'only' asks for it.
  EXPECT_TRUE(filterSignatures(input, "min", {"min"}).empty());
  EXPECT_EQ(filterSignatures(input, "min,sum", {"min"}).size(), 1);
}

} // namespace facebook::velox::fuzzer::test
