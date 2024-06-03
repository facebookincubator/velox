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

#include "velox/exec/Aggregate.h"
#include "velox/exec/tests/SimpleAggregateFunctionsRegistration.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using facebook::velox::functions::aggregate::test::AggregationTestBase;

namespace facebook::velox::aggregate::test {
namespace {

class SimpleSumAggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();

    registerSimpleSumAggregate("simple_sum");
  }
};

TEST_F(SimpleSumAggregationTest, basic) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeFlatVector<bool>({true, true, false, false}),
  });
  auto expected = makeRowVector(
      {makeFlatVector<bool>({false, true}), makeFlatVector<int64_t>({-7, -3})});
  testAggregations({data}, {"c1"}, {"simple_sum(c0)"}, {expected});
}

} // namespace
} // namespace facebook::velox::aggregate::test
