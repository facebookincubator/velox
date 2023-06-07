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
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/AggregationTestBase.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {
// Helper generates aggregation over column string.
std::string genAggr(const char* aggrName, const char* colName) {
  return fmt::format("{}({})", aggrName, colName);
}

// Macro to make it even shorter (assumes we have 'aggrName' var on the stack).
#define GEN_AGG(_colName_) genAggr(aggrName, _colName_)

// The test class.
class EntropyAggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
  }
};

TEST_F(EntropyAggregationTest, entropyConst) {
  auto expectedValue = 5.0 / 100 * std::log2(100 / 5) * 20;
  auto data = {
      makeRowVector({makeConstant(5, 10)}),
      makeRowVector({makeConstant(5, 10)}),
  };
  std::vector<double> expectedVec = {expectedValue};
  auto expectedResult = makeRowVector({makeFlatVector<double>(expectedVec)});
  testAggregations(data, {}, {"entropy(c0)"}, {expectedResult});
}

} // namespace
} // namespace facebook::velox::aggregate::test
