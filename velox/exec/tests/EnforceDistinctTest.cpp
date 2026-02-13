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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::exec {
namespace {

class EnforceDistinctTest : public OperatorTestBase {
 protected:
  core::PlanNodePtr makePlan(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& keys,
      const std::string& errorMessage) {
    return PlanBuilder()
        .values(input)
        .enforceDistinct(keys, errorMessage)
        .planNode();
  }

  core::PlanNodePtr makePlan(
      const RowVectorPtr& input,
      const std::string& key,
      const std::string& errorMessage) {
    return makePlan(
        std::vector<RowVectorPtr>{input},
        std::vector<std::string>{key},
        errorMessage);
  }

  void assertDistinct(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& keys) {
    auto plan = makePlan(input, keys, "Duplicate key found");
    AssertQueryBuilder(plan).assertResults(input);
  }
};

TEST_F(EnforceDistinctTest, uniqueRowsSingleKey) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>({1, 2, 3, std::nullopt, 5, 6, 7, 8, 9}),
      makeFlatVector<std::string>(
          {"a", "a", "b", "b", "a", "a", "b", "b", "a"}),
  });

  assertDistinct(split(data, 3), {"c0"});
}

TEST_F(EnforceDistinctTest, uniqueRowsMultipleKeys) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3, 4, 4, 5}),
      makeFlatVector<std::string>(
          {"x", "x", "y", "y", "x", "x", "y", "y", "x"}),
      makeFlatVector<int64_t>({10, 20, 10, 20, 10, 20, 10, 20, 10}),
  });

  assertDistinct(split(data, 3), {"c0", "c2"});
}

TEST_F(EnforceDistinctTest, duplicateWithinBatch) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 2, 5}),
  });

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(makePlan(data, "c0", "Duplicate key found"))
          .countResults(),
      "Duplicate key found");
}

TEST_F(EnforceDistinctTest, duplicateAcrossBatches) {
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
  });

  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({4, 2, 6}),
  });

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(
          makePlan({batch1, batch2}, {"c0"}, "Duplicate key found"))
          .countResults(),
      "Duplicate key found");
}

TEST_F(EnforceDistinctTest, emptyInput) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({}),
  });

  assertDistinct({data}, {"c0"});
}

TEST_F(EnforceDistinctTest, singleRow) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({42}),
  });

  assertDistinct({data}, {"c0"});
}

TEST_F(EnforceDistinctTest, duplicateNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>({1, std::nullopt, 3, std::nullopt, 5}),
  });

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(makePlan(data, "c0", "Duplicate key found"))
          .countResults(),
      "Duplicate key found");
}

} // namespace
} // namespace facebook::velox::exec
