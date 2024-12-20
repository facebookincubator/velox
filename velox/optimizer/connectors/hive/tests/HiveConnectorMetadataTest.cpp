
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

#include "velox/exec/tests/utils/DistributedPlanBuilder.h"
#include "velox/exec/tests/utils/LocalRunnerTestBase.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/optimizer/connectors/hive/LocalHiveConnectorMetadata.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector;

class HiveConnectorMetadataTest : public LocalRunnerTestBase {
 protected:
  static constexpr int32_t kNumFiles = 5;
  static constexpr int32_t kNumVectors = 5;
  static constexpr int32_t kRowsPerVector = 10000;
  static constexpr int32_t kNumRows = kNumFiles * kNumVectors * kRowsPerVector;

  static void SetUpTestCase() {
    // The lambdas will be run after this scope returns, so make captures
    // static.
    static int32_t counter1;
    // Clear 'counter1' so that --gtest_repeat runs get the same data.
    counter1 = 0;
    auto customize1 = [&](const RowVectorPtr& rows) {
      makeAscending(rows, counter1);
    };

    rowType_ = ROW({"c0"}, {BIGINT()});
    testTables_ = {TableSpec{
        .name = "T",
        .columns = rowType_,
        .rowsPerVector = kRowsPerVector,
        .numVectorsPerFile = kNumVectors,
        .numFiles = kNumFiles,
        .customizeData = customize1}};

    // Creates the data and schema from 'testTables_'. These are created on the
    // first test fixture initialization.
    LocalRunnerTestBase::SetUpTestCase();
  }

  static void makeAscending(const RowVectorPtr& rows, int32_t& counter) {
    auto ints = rows->childAt(0)->as<FlatVector<int64_t>>();
    for (auto i = 0; i < ints->size(); ++i) {
      ints->set(i, counter + i);
    }
    counter += ints->size();
  }

  inline static RowTypePtr rowType_;
};

TEST_F(HiveConnectorMetadataTest, basic) {
  auto connector = getConnector(kHiveConnectorId);
  auto metadata = connector->metadata();
  ASSERT_TRUE(metadata != nullptr);
  auto table = metadata->findTable("T");
  ASSERT_TRUE(table != nullptr);
  auto column = table->findColumn("c0");
  ASSERT_TRUE(column != nullptr);
  EXPECT_EQ(250'000, table->numRows());
  auto* layout = table->layouts()[0];
  auto columnHandle = metadata->createColumnHandle(*layout, "c0");
  std::vector<ColumnHandlePtr> columns = {columnHandle};
  std::vector<core::TypedExprPtr> filters;
  std::vector<core::TypedExprPtr> rejectedFilters;
  auto ctx = dynamic_cast<hive::LocalHiveConnectorMetadata*>(metadata)
                 ->connectorQueryCtx();

  auto tableHandle = metadata->createTableHandle(
      *layout, columns, *ctx->expressionEvaluator(), filters, rejectedFilters);
  EXPECT_TRUE(rejectedFilters.empty());
  std::vector<ColumnStatistics> stats;
  std::vector<common::Subfield> fields;
  auto c0 = common::Subfield::create("c0");
  fields.push_back(std::move(*c0));
  HashStringAllocator allocator(pool_.get());
  auto pair = layout->sample(tableHandle, 100, {}, fields, &allocator, &stats);
  EXPECT_EQ(250'000, pair.first);
  EXPECT_EQ(250'000, pair.second);
}
