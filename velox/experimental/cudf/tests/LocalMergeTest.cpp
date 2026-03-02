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

#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <fmt/format.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace {

class LocalMergeTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void testSingleKey(
      const std::vector<RowVectorPtr>& inputVectors,
      const std::string& key) {
    auto keyIndex = inputVectors[0]->type()->asRow().getChildIdx(key);

    std::vector<std::string> sortOrderSqls = {
        "NULLS LAST", "NULLS FIRST", "DESC NULLS FIRST", "DESC NULLS LAST"};
    for (const auto& sortOrderSql : sortOrderSqls) {
      const auto orderByClause = fmt::format("{} {}", key, sortOrderSql);
      auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
      auto plan = PlanBuilder(planNodeIdGenerator)
                      .localMerge(
                          {orderByClause},
                          {PlanBuilder(planNodeIdGenerator)
                               .values(inputVectors)
                               .orderBy({orderByClause}, true)
                               .planNode()})
                      .planNode();
      CursorParameters params;
      params.planNode = plan;
      params.maxDrivers = 1;
      assertQueryOrdered(
          params,
          "SELECT * FROM (SELECT * FROM tmp) ORDER BY " + orderByClause,
          {keyIndex});

      std::vector<std::shared_ptr<const core::PlanNode>> sources;
      for (const auto& input : inputVectors) {
        sources.push_back(PlanBuilder(planNodeIdGenerator)
                              .values({input})
                              .orderBy({orderByClause}, true)
                              .planNode());
      }
      plan = PlanBuilder(planNodeIdGenerator)
                 .localMerge({orderByClause}, std::move(sources))
                 .planNode();

      assertQueryOrdered(
          plan, "SELECT * FROM tmp ORDER BY " + orderByClause, {keyIndex});
    }
  }

  void testTwoKeys(
      const std::vector<RowVectorPtr>& inputVectors,
      const std::string& key1,
      const std::string& key2) {
    auto& rowType = inputVectors[0]->type()->asRow();
    auto sortingKeys = {rowType.getChildIdx(key1), rowType.getChildIdx(key2)};

    std::vector<core::SortOrder> sortOrders = {
        core::kAscNullsLast,
        core::kAscNullsFirst,
        core::kDescNullsFirst,
        core::kDescNullsLast};
    std::vector<std::string> sortOrderSqls = {
        "NULLS LAST", "NULLS FIRST", "DESC NULLS FIRST", "DESC NULLS LAST"};

    for (auto i = 0; i < sortOrders.size(); ++i) {
      for (auto j = 0; j < sortOrders.size(); ++j) {
        const std::vector<std::string> orderByClauses = {
            fmt::format("{} {}", key1, sortOrderSqls[i]),
            fmt::format("{} {}", key2, sortOrderSqls[j])};
        const auto orderBySql = fmt::format(
            "ORDER BY {}, {}", orderByClauses[0], orderByClauses[1]);
        auto planNodeIdGenerator =
            std::make_shared<core::PlanNodeIdGenerator>();
        auto plan = PlanBuilder(planNodeIdGenerator)
                        .localMerge(
                            orderByClauses,
                            {PlanBuilder(planNodeIdGenerator)
                                 .values(inputVectors)
                                 .orderBy(orderByClauses, true)
                                 .planNode()})
                        .planNode();
        CursorParameters params;
        params.planNode = plan;
        params.maxDrivers = 1;
        assertQueryOrdered(
            params,
            "SELECT * FROM (SELECT * FROM tmp) " + orderBySql,
            sortingKeys);

        std::vector<std::shared_ptr<const core::PlanNode>> sources;
        for (const auto& input : inputVectors) {
          sources.push_back(PlanBuilder(planNodeIdGenerator)
                                .values({input})
                                .orderBy(orderByClauses, true)
                                .planNode());
        }
        plan = PlanBuilder(planNodeIdGenerator)
                   .localMerge(orderByClauses, std::move(sources))
                   .planNode();

        assertQueryOrdered(
            plan, "SELECT * FROM tmp " + orderBySql, sortingKeys);
      }
    }
  }
};

TEST_F(LocalMergeTest, localMerge) {
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 3; ++i) {
    static constexpr vector_size_t kBatchSize = 100;
    auto c0 = makeFlatVector<int64_t>(
        kBatchSize,
        [&](auto row) { return kBatchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<int64_t>(
        kBatchSize, [&](auto row) { return row; }, nullEvery(5));
    auto c2 = makeFlatVector<double>(
        kBatchSize, [](auto row) { return row * 0.1; }, nullEvery(11));
    auto c3 = makeFlatVector<StringView>(kBatchSize, [](auto row) {
      return StringView::makeInline(std::to_string(row));
    });
    vectors.push_back(makeRowVector({c0, c1, c2, c3}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c0");
  testSingleKey(vectors, "c3");

  testTwoKeys(vectors, "c0", "c3");
  testTwoKeys(vectors, "c3", "c0");
}

TEST_F(LocalMergeTest, offByOne) {
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 10}),
  });

  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({2, 3, 4, 5}),
  });

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localMerge(
              {"c0"},
              {
                  PlanBuilder(planNodeIdGenerator).values({data1}).planNode(),
                  PlanBuilder(planNodeIdGenerator).values({data2}).planNode(),
              })
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  params.queryCtx = core::QueryCtx::create(executor_.get());
  params.queryCtx->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kPreferredOutputBatchRows, "6"}});
  assertQueryOrdered(params, "VALUES (0), (1), (2), (3), (4), (5), (10)", {0});
}

} // namespace
