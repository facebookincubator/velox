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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::connector::hive::iceberg::test {

class IcebergWriterModeTest : public IcebergTestBase,
                              public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();

    std::unordered_map<std::string, std::string> sessionProps = {
        {HiveConfig::kFanoutEnabledSession, GetParam() ? "true" : "false"},
    };

    connectorSessionProperties_ =
        std::make_shared<config::ConfigBase>(std::move(sessionProps), true);

    setupMemoryPools("IcebergWriterModeTest");
  }
};

INSTANTIATE_TEST_SUITE_P(
    FanoutModes,
    IcebergWriterModeTest,
    ::testing::Values(true, false),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "FanoutEnabled" : "FanoutDisabled";
    });

TEST_P(IcebergWriterModeTest, identityPartitioning) {
  constexpr auto size = 10;
  std::vector<std::string> names = {"c_int"};
  std::vector<TypePtr> types = {INTEGER()};
  rowType_ = ROW(names, types);

  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_,
      outputDirectory->getPath(),
      {{0, TransformType::kIdentity, std::nullopt}});

  auto intVector1 =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; });
  auto vector1 = makeRowVector(names, {intVector1});
  auto intVector2 =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row + 10; });
  auto vector2 = makeRowVector(names, {intVector2});
  dataSink->appendData(vector1);
  dataSink->appendData(vector2);
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  createDuckDbTable({vector1, vector2});
  auto splits = createSplitsForDirectory(outputDirectory->getPath());
  auto plan = exec::test::PlanBuilder().tableScan(rowType_).planNode();
  assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
}

TEST_P(IcebergWriterModeTest, clusteredInput) {
  constexpr auto size = 100;
  std::vector<std::string> names = {"c_int"};
  std::vector<TypePtr> types = {INTEGER()};
  rowType_ = ROW(names, types);

  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_,
      outputDirectory->getPath(),
      {{0, TransformType::kIdentity, std::nullopt}});

  auto intVector1 = makeConstant(100, size, INTEGER());
  auto vector1 = makeRowVector(names, {intVector1});
  auto intVector2 = makeConstant(100, size, INTEGER());
  auto vector2 = makeRowVector(names, {intVector2});
  dataSink->appendData(vector1);
  dataSink->appendData(vector2);
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  auto stats = dataSink->dataFileStats();
  ASSERT_EQ(stats.at(0)->numRecords, size * 2);
  ASSERT_FALSE(stats.at(0)->lowerBounds.empty());
  ASSERT_FALSE(stats.at(0)->upperBounds.empty());
  createDuckDbTable({vector1, vector2});
  auto splits = createSplitsForDirectory(outputDirectory->getPath());
  ASSERT_EQ(splits.size(), 1);
  auto plan = exec::test::PlanBuilder().tableScan(rowType_).planNode();
  assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
}

TEST_P(IcebergWriterModeTest, clusteredNullInput) {
  constexpr auto size = 100;
  std::vector<std::string> names = {"c_int"};
  std::vector<TypePtr> types = {INTEGER()};
  rowType_ = ROW(names, types);

  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_,
      outputDirectory->getPath(),
      {{0, TransformType::kIdentity, std::nullopt}});

  auto intVector1 = makeNullConstant(TypeKind::INTEGER, size);
  auto vector1 = makeRowVector(names, {intVector1});
  auto intVector2 = makeNullConstant(TypeKind::INTEGER, size);
  auto vector2 = makeRowVector(names, {intVector2});
  dataSink->appendData(vector1);
  dataSink->appendData(vector2);
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  auto stats = dataSink->dataFileStats();
  ASSERT_TRUE(stats.at(0)->upperBounds.empty());
  ASSERT_EQ(stats.at(0)->nullValueCounts.at(1), size * 2);
  createDuckDbTable({vector1, vector2});
  auto splits = createSplitsForDirectory(outputDirectory->getPath());
  ASSERT_EQ(splits.size(), 1);
  auto plan = exec::test::PlanBuilder().tableScan(rowType_).planNode();
  assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
}

TEST_P(IcebergWriterModeTest, sortedByAndIdentityPartittioning) {
  constexpr auto size = 10;
  std::vector<std::string> names = {"c_int"};
  std::vector<TypePtr> types = {INTEGER()};
  rowType_ = ROW(names, types);

  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_,
      outputDirectory->getPath(),
      {{0, TransformType::kIdentity, std::nullopt}},
      {"c_int DESC"});

  auto intVector1 =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; });
  auto vector1 = makeRowVector(names, {intVector1});
  auto intVector2 =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row + 10; });
  auto vector2 = makeRowVector(names, {intVector2});
  dataSink->appendData(vector1);
  dataSink->appendData(vector2);
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
  createDuckDbTable({vector1, vector2});
  auto splits = createSplitsForDirectory(outputDirectory->getPath());
  ASSERT_EQ(splits.size(), size * 2);
  auto plan = exec::test::PlanBuilder().tableScan(rowType_).planNode();
  assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
}

TEST_P(IcebergWriterModeTest, nonClusteredInput) {
  constexpr auto size = 10;
  std::vector<std::string> names = {"c_int"};
  std::vector<TypePtr> types = {INTEGER()};
  rowType_ = ROW(names, types);

  auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_,
      outputDirectory->getPath(),
      {{0, TransformType::kIdentity, std::nullopt}});

  auto intVector1 =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; });
  auto vector1 = makeRowVector(names, {intVector1});
  auto intVector2 =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row + 5; });
  auto vector2 = makeRowVector(names, {intVector2});
  dataSink->appendData(vector1);
  if (!GetParam()) {
    VELOX_ASSERT_THROW(
        dataSink->appendData(vector2),
        "Incoming records violate the writer assumption that records are clustered by spec and \n by partition within each spec. Either cluster the incoming records or switch to fanout writers.\nEncountered records that belong to already closed files:\n");
  } else {
    dataSink->appendData(vector2);
    ASSERT_TRUE(dataSink->finish());
    dataSink->close();
    createDuckDbTable({vector1, vector2});
    auto splits = createSplitsForDirectory(outputDirectory->getPath());
    auto plan = exec::test::PlanBuilder().tableScan(rowType_).planNode();
    assertQuery(plan, splits, fmt::format("SELECT * FROM tmp"));
  }
}

} // namespace facebook::velox::connector::hive::iceberg::test
